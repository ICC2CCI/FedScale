#!/bin/bash
# 多机多卡分布式训练 - 自动化部署脚本
# 使用方法: bash deploy_multinode.sh

set -e

echo "=========================================="
echo "多机多卡分布式训练 - 自动化部署"
echo "=========================================="
echo ""

# 配置变量
IMAGE_NAME="ccr.ccs.tencentyun.com/czfflower/llama_3b_v2-client-multinode"
IMAGE_TAG="latest"
NAMESPACE="flower-supernode-a"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 打印函数
print_step() {
    echo -e "${GREEN}[步骤 $1]${NC} $2"
}

print_warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

print_error() {
    echo -e "${RED}[错误]${NC} $1"
}

# Step 0: 前置检查
print_step "0" "检查前置条件..."

# 检查kubectl
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl 未安装，请先安装kubectl"
    exit 1
fi

# 检查docker
if ! command -v docker &> /dev/null; then
    print_error "docker 未安装，请先安装docker"
    exit 1
fi

# 检查命名空间
if ! kubectl get namespace $NAMESPACE &> /dev/null; then
    print_warning "命名空间 $NAMESPACE 不存在，将自动创建"
    kubectl create namespace $NAMESPACE
fi

echo "✓ 前置检查通过"
echo ""

# Step 1: 构建Docker镜像
print_step "1" "构建Docker镜像..."

cd flowertune-llm

if [ -f "Dockerfile.multinode" ]; then
    echo "正在构建镜像: ${IMAGE_NAME}:${IMAGE_TAG}"
    docker build -f Dockerfile.multinode -t ${IMAGE_NAME}:${IMAGE_TAG} .
    
    if [ $? -eq 0 ]; then
        echo "✓ 镜像构建成功"
    else
        print_error "镜像构建失败"
        exit 1
    fi
else
    print_error "Dockerfile.multinode 不存在"
    exit 1
fi

echo ""

# Step 2: 推送镜像（可选）
read -p "是否推送镜像到仓库? (y/n): " PUSH_IMAGE
if [ "$PUSH_IMAGE" = "y" ] || [ "$PUSH_IMAGE" = "Y" ]; then
    print_step "2" "推送Docker镜像..."
    docker push ${IMAGE_NAME}:${IMAGE_TAG}
    echo "✓ 镜像推送成功"
else
    print_warning "跳过镜像推送"
fi

echo ""

# Step 3: 配置RBAC权限
print_step "3" "配置RBAC权限..."

cd ..

if [ -f "configs/flower-client-rbac.yaml" ]; then
    kubectl apply -f configs/flower-client-rbac.yaml
    
    if [ $? -eq 0 ]; then
        echo "✓ RBAC配置成功"
    else
        print_error "RBAC配置失败"
        exit 1
    fi
else
    print_error "flower-client-rbac.yaml 不存在"
    exit 1
fi

echo ""

# Step 4: 验证RBAC权限
print_step "4" "验证RBAC权限..."

CAN_CREATE_JOBS=$(kubectl auth can-i create jobs \
    --as=system:serviceaccount:${NAMESPACE}:flower-client-sa \
    -n ${NAMESPACE})

if [ "$CAN_CREATE_JOBS" = "yes" ]; then
    echo "✓ RBAC权限验证通过"
else
    print_error "RBAC权限验证失败"
    exit 1
fi

echo ""

# Step 5: 检查PVC
print_step "5" "检查共享存储PVC..."

CACHE_PVC_EXISTS=$(kubectl get pvc flowertune-cache-pvc -n ${NAMESPACE} --ignore-not-found)
OUTPUT_PVC_EXISTS=$(kubectl get pvc flowertune-output-pvc -n ${NAMESPACE} --ignore-not-found)

if [ -z "$CACHE_PVC_EXISTS" ]; then
    print_warning "flowertune-cache-pvc 不存在，请手动创建"
fi

if [ -z "$OUTPUT_PVC_EXISTS" ]; then
    print_warning "flowertune-output-pvc 不存在，请手动创建"
fi

if [ -n "$CACHE_PVC_EXISTS" ] && [ -n "$OUTPUT_PVC_EXISTS" ]; then
    echo "✓ PVC检查通过"
else
    read -p "PVC未完全就绪，是否继续部署? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        exit 1
    fi
fi

echo ""

# Step 6: 配置部署参数
print_step "6" "配置部署参数..."

read -p "节点数量 (默认2): " NUM_NODES
NUM_NODES=${NUM_NODES:-2}

read -p "每节点GPU数 (默认4): " GPUS_PER_NODE
GPUS_PER_NODE=${GPUS_PER_NODE:-4}

TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
echo "总GPU数: $TOTAL_GPUS"

# 更新配置文件
if [ -f "configs/clientapp-multinode-deployment.yaml" ]; then
    sed -i "s/value: \"[0-9]*\"  # Number of worker nodes/value: \"$NUM_NODES\"  # Number of worker nodes/" configs/clientapp-multinode-deployment.yaml
    sed -i "s/value: \"[0-9]*\"  # GPUs per worker node/value: \"$GPUS_PER_NODE\"  # GPUs per worker node/" configs/clientapp-multinode-deployment.yaml
    echo "✓ 配置已更新: $NUM_NODES 节点 × $GPUS_PER_NODE GPU = $TOTAL_GPUS 总GPU"
else
    print_error "clientapp-multinode-deployment.yaml 不存在"
    exit 1
fi

echo ""

# Step 7: 部署Client
print_step "7" "部署Flower Client..."

kubectl apply -f configs/clientapp-multinode-deployment.yaml

if [ $? -eq 0 ]; then
    echo "✓ Client部署成功"
else
    print_error "Client部署失败"
    exit 1
fi

echo ""

# Step 8: 验证部署
print_step "8" "验证部署状态..."

sleep 5

POD_STATUS=$(kubectl get pods -n ${NAMESPACE} -l app=clientapp-multinode -o jsonpath='{.items[0].status.phase}')

if [ "$POD_STATUS" = "Running" ]; then
    echo "✓ Client Pod运行正常"
else
    print_warning "Client Pod状态: $POD_STATUS"
    echo "请使用以下命令查看详细信息:"
    echo "  kubectl describe pod -n ${NAMESPACE} -l app=clientapp-multinode"
    echo "  kubectl logs -n ${NAMESPACE} -l app=clientapp-multinode"
fi

echo ""

# 完成
echo "=========================================="
echo -e "${GREEN}✓ 部署完成！${NC}"
echo "=========================================="
echo ""
echo "后续步骤:"
echo "1. 监控Client日志:"
echo "   kubectl logs -f deployment/clientapp-multinode -n ${NAMESPACE}"
echo ""
echo "2. 查看训练Jobs:"
echo "   kubectl get jobs -n ${NAMESPACE}"
echo ""
echo "3. 查看详细文档:"
echo "   cat MULTINODE_DEPLOYMENT_GUIDE.md"
echo ""
echo "=========================================="
