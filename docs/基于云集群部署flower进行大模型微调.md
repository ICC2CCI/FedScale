跨云集群 Skupper + Flower 部署方案（已成功部署flower框架）

---

## 前置约定

| 角色 | 环境 | 配置 |
|---|---|---|
| 中心服务器集群center | 腾讯云 TKE | kubeconfig 已配置为 `~/.kube/config-center` |
| 集群 A | 腾讯云 TKE | kubeconfig 已配置为 `~/.kube/config-tke` |
| 集群 B | 阿里云 ACK | kubeconfig 已配置为 `~/.kube/config-ack` |

Flower SuperLink 端口约定：`9092`（Fleet API，SuperNode 连接用）、`9093`（Control API，ServerApp 连接用）。
详见[官网说明](https://flower.ai/docs/framework/ref-flower-network-communication.html)

Flower框架结构如图：
![flower架构图](https://flower.ai/docs/framework/_static/flower-network-diagram-subprocess-light.svg)

以下所有配置均可在远程主机使用kubectl和skupper cli进行，无需ssh进入集群中节点服务器

---

## 阶段一：环境准备

### 1.1 安装 Skupper CLI
Skupper CLI是用于部署、配置Skupper的工具，可以装在3个集群之外的远程主机上。
本方案中装在远程主机上，系统为Ubuntu24.04

```bash
# Linux x86_64
curl https://skupper.io/install.sh | sh

skupper version  # 确认安装成功
```

### 1.2 安装kubectl
kubectl用于远程控制集群
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
```
参考[官网](https://kubernetes.io/zh-cn/docs/tasks/tools/)

### 1.3 配置三份 kubeconfig

```bash
export KUBECONFIG_CENTER=~/.kube/config-center   # 中心服务器 k8s
export KUBECONFIG_TKE=~/.kube/config-tke         # 腾讯云 TKE
export KUBECONFIG_ACK=~/.kube/config-ack         # 阿里云 ACK

# 验证三个集群连通
kubectl --kubeconfig=$KUBECONFIG_CENTER get nodes
kubectl --kubeconfig=$KUBECONFIG_TKE   get nodes
kubectl --kubeconfig=$KUBECONFIG_ACK   get nodes
```

### 1.4 各集群创建专用 Namespace

```bash
# 中心服务器
kubectl --kubeconfig=$KUBECONFIG_CENTER create namespace flower-superlink

# TKE 集群 A
kubectl --kubeconfig=$KUBECONFIG_TKE create namespace flower-supernode-a

# ACK 集群 B
kubectl --kubeconfig=$KUBECONFIG_ACK create namespace flower-supernode-b
```

### 1.4 在3个集群中安装skupper

```bash
# 中心服务器
kubectl --kubeconfig=$KUBECONFIG_CENTER apply -f https://skupper.io/install.yaml

# TKE 集群 A
kubectl --kubeconfig=$KUBECONFIG_TKE apply -f https://skupper.io/install.yaml

# ACK 集群 B
kubectl --kubeconfig=$KUBECONFIG_ACK apply -f https://skupper.io/install.yaml
```


---

## 阶段二：部署 Skupper 并建立连接

### 2.1 在中心服务器集群初始化 Skupper Site（启用对外链接接入）

```bash
# 切换到中心服务器 kubeconfig
export KUBECONFIG=$KUBECONFIG_CENTER

# 在 flower-superlink namespace 创建 Skupper Site
skupper site create superlink-site -n flower-superlink --enable-link-access

# 等待 Site 就绪
skupper site status -n flower-superlink
# 期望输出：Site "superlink-site" is ready.
```

此时 Skupper 会在 `flower-superlink` namespace 创建一个 `skupper-router` Deployment，并通过 LoadBalancer Service 暴露一个公网端点

```bash
# 可以查看 Skupper 对外暴露的端点
kubectl --kubeconfig=$KUBECONFIG_CENTER \
  get svc -n flower-superlink skupper-router
# NAME             TYPE           CLUSTER-IP    EXTERNAL-IP    PORT(S)
# skupper-router   LoadBalancer   x.x.x.x       x.x.x.x        55671:xxxxx/TCP
```

### 2.2 中心服务器生成 skupper link 配置文件（供 TKE 和 ACK 使用）

```bash
export KUBECONFIG=$KUBECONFIG_CENTER

skupper link generate > superlink.yaml --namespace flower-superlink
```


### 2.3 在 TKE 集群 A 初始化 Skupper Site 并接入网络

```bash
export KUBECONFIG=$KUBECONFIG_TKE

# 初始化 Site
skupper site create supernode-site-a --namespace flower-supernode-a

# 建立到中心服务器 Site 的 link
kubectl apply -f superlink.yaml --namespace flower-supernode-a

# 验证 link 状态
skupper link status --namespace flower-supernode-a
# 期望输出：
# NAME                  STATUS  COST  MESSAGE
# link-to-superlink     Ready   1     OK
```

### 2.4 在 ACK 集群 B 初始化 Skupper Site 并接入网络

```bash
export KUBECONFIG=$KUBECONFIG_ACK

skupper site create supernode-site-b --namespace flower-supernode-b

kubectl apply -f superlink.yaml --namespace flower-supernode-b

skupper link status --namespace flower-supernode-b
# 期望输出：link-to-superlink   Ready
```

---


## 阶段三：部署 Flower SuperLink 和 SuperNode

SuperLink 使用官方 Docker 镜像，部署在中心服务器的 K8s 集群中。

### 3.1 部署 SuperLink Deployment
创建文件superlink-deployment.yaml
复制下面内容进入文件
```yaml
# superlink-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: superlink
  namespace: flower-superlink # namespace要匹配
spec:
  replicas: 1
  selector:
    matchLabels:
      app: superlink
  template:
    metadata:
      labels:
        app: superlink
    spec:
      containers:
        - name: superlink
          image: flwr/superlink:1.28.0 # 镜像拉取失败的话可以改为docker.1ms.run/flwr/superlink:1.28.0
          args:
            - "--insecure"
            - "--isolation" # 用于独立创建superexec，不加该参数则表示superexec服务由superlink自动创建为子进程
          ports: # which ports to expose/available
            - containerPort: 9091
              name: superexec
            - containerPort: 9092
              name: fleet
            - containerPort: 9093
              name: control
          volumeMounts:
            - name: cache-volume
              mountPath: /app/.cache
            - name: tmp-volume
              mountPath: /var/tmp
            - name: fab-volume
              mountPath: /app/.flwr
            - name: config-volume
              mountPath: /app/.config
      volumes:
        - name: cache-volume
          emptyDir:
            sizeLimit: 50Mi
        - name: tmp-volume
          emptyDir:
            sizeLimit: 50Mi
        - name: fab-volume
          emptyDir:
            sizeLimit: 50Mi
        - name: config-volume
          emptyDir:
            sizeLimit: 50Mi
```
在中心服务器集群上使用yaml文件部署
```bash
kubectl --kubeconfig=$KUBECONFIG_CENTER -n flower-superlink apply -f superlink-deployment.yaml

# 验证 SuperLink 启动
kubectl --kubeconfig=$KUBECONFIG_CENTER -n flower-superlink logs -l app=superlink --tail=20
```

### 3.2 为 SuperLink 创建 K8s Service

Skupper Connector 需要通过 K8s Service 或直接绑定 Deployment 来发现 Pod，此处创建 ClusterIP Service：

```yaml
# superlink-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: superlink-service
  namespace: flower-superlink
spec:
  selector:
    app: superlink
  ports:  # like a dynamic IP routing table/mapping that routes traffic to the designated ports
  - protocol: TCP
    port: 9092   # Port for SuperNode connection
    targetPort: fleet  # the SuperLink container port
    name: superlink-fleetapi
  - protocol: TCP
    port: 9093   # Port for Flower app submission
    targetPort: control  # the SuperLink container port
    name: superlink-controlapi
  type: LoadBalancer  # balances workload, makes the service publicly available
```
在中心服务器集群上使用yaml文件部署
```bash
kubectl --kubeconfig=$KUBECONFIG_CENTER apply -f superlink-service.yaml
```

### 3.3 部署 SuperNode A（TKE）
创建supernode-a-deployment.yaml文件，并复制下面内容：
```yaml
# supernode-a-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: supernode-a
  namespace: flower-supernode-a
spec:
  replicas: 1
  selector:
    matchLabels:
      app: supernode-a
  template:
    metadata:
      labels:
        app: supernode-a
    spec:
      containers:
      - name: supernode
        image: flwr/supernode:1.28.0
        args:
          - "--insecure"
          - "--superlink"
          - "fleet-api:9092" # Use the listener service DNS
          - "--clientappio-api-address"
          - "0.0.0.0:9094"
          - "--node-config" # partition-id表示该node在所有node中的序号，num-partitions表示一共有多少个可用node
          - "partition-id=0 num-partitions=2"
          - "--isolation" # 该参数表示superexec由我们手动独立创建，不加该参数则表示有supernode服务创建子进程
          - "process"
        ports:
        - containerPort: 9094
        volumeMounts:
        - name: cache-volume
          mountPath: /app/.cache
        - name: tmp-volume
          mountPath: /var/tmp
        - name: fab-volume
          mountPath: /app/.flwr
        - name: config-volume
          mountPath: /app/.config
      volumes:
      - name: cache-volume
        emptyDir:
          sizeLimit: 50Mi
      - name: tmp-volume
        emptyDir:
          sizeLimit: 50Mi
      - name: fab-volume
        emptyDir:
          sizeLimit: 50Mi
      - name: config-volume
        emptyDir:
          sizeLimit: 50Mi
```

```bash
kubectl --kubeconfig=$KUBECONFIG_TKE -n flower-supernode-a apply -f supernode-a-deployment.yaml

# 查看 SuperNode A 日志，此时还无法连接成功，还需要通过skupper建立连接
```

### 3.4 部署 SuperNode B（ACK）
创建supernode-b-deployment.yaml文件，并复制下面内容：
```yaml
# supernode-b-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: supernode-b
  namespace: flower-supernode-b
spec:
  replicas: 1
  selector:
    matchLabels:
      app: supernode-b
  template:
    metadata:
      labels:
        app: supernode-b
    spec:
      containers:
      - name: supernode
        image: flwr/supernode:1.28.0
        args:
          - "--insecure"
          - "--superlink"
          - "fleet-api:9092" # Use the listener service DNS
          - "--clientappio-api-address"
          - "0.0.0.0:9094"
          - "--node-config" # partition-id表示该node在所有node中的序号，num-partitions表示一共有多少个可用node
          - "partition-id=1 num-partitions=2"
          - "--isolation" # 该参数表示superexec由我们手动独立创建，不加该参数则表示有supernode服务创建子进程
          - "process"
        ports:
        - containerPort: 9094
        volumeMounts:
        - name: cache-volume
          mountPath: /app/.cache
        - name: tmp-volume
          mountPath: /var/tmp
        - name: fab-volume
          mountPath: /app/.flwr
        - name: config-volume
          mountPath: /app/.config
      volumes:
      - name: cache-volume
        emptyDir:
          sizeLimit: 50Mi
      - name: tmp-volume
        emptyDir:
          sizeLimit: 50Mi
      - name: fab-volume
        emptyDir:
          sizeLimit: 50Mi
      - name: config-volume
        emptyDir:
          sizeLimit: 50Mi
```
由于在TKE和ACK两个集群上启动了两个supernode，因此num-partitions参数为2

```bash
kubectl --kubeconfig=$KUBECONFIG_ACK apply -f supernode-b-deployment.yaml

# 查看 SuperNode B 日志，此时还无法连接成功，还需要通过skupper建立连接
```

---


## 阶段四：通过Skupper建立flower superlink 和 supernode 连接

### 4.1 在中心服务器 Site 创建 Connector（暴露 SuperLink 服务）

Connector 的作用是把 SuperLink 的 K8s Service 暴露到 Skupper 应用网络，让 TKE/ACK 的 SuperNode 可以通过虚拟 DNS 名访问。

```bash
export KUBECONFIG=$KUBECONFIG_CENTER

# 创建 connector，绑定到 superlink deployment，暴露 Fleet API 端口 9092
skupper connector create fleet-api 9092 --namespace flower-superlink --workload deployment/superlink

# 此处的deployment/superlink与第三部分部署superlink服务时名字对应
```

### 4.2 在 TKE 和 ACK 创建 Listener（接收来自 Skupper 网络的服务）

```bash
# TKE 集群 A
export KUBECONFIG=$KUBECONFIG_TKE
skupper listener create fleet-api 9092 --namespace flower-supernode-a
# Skupper 自动创建一个名为 fleet-api 的 K8s Service，SuperNode 连接这个 Service

# ACK 集群 B
export KUBECONFIG=$KUBECONFIG_ACK
skupper listener create fleet-api 9092 --namespace flower-supernode-b
```

此时验证 superlink 和 supernode 连接则应显示成功，如不成功，重启两个supernode对应的pod
重启方法：直接删除pod，由于k8s机制，deployment会自动生成新的pod。
验证：
```bash
kubectl --kubeconfig=$KUBECONFIG_TKE -n flower-supernode-a logs -l app=supernode-a --tail=20
# ...
# INFO :      SuperNode ID: xxxxxxxxx
# ...

kubectl --kubeconfig=$KUBECONFIG_ACK -n flower-supernode-b logs -l app=supernode-b --tail=20
# ...
# INFO :      SuperNode ID: xxxxxxxxx
# ...
```

---


## 阶段五：建立服务端和客户端的superexec
### 5.1 制作镜像
superexec服务所需的docker镜像根据执行任务的不同有所区别，需要单独定制，没有官方统一版本。
```Dockerfile
FROM flwr/superexec:1.28.0

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    meson \
    ninja-build \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
   && python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flower-superexec"]
```
1. 其中gfortran、meson等是编译相关工具，由于openllama微调项目所需的一些python库编译需要，因此也要制作到镜像中，基础镜像源flwr/superexec:1.28.0是官方镜像
2. 制作时需要将该文件放到flower项目根目录下，与项目pyproject.toml文件同一目录。pyproject.toml是根据项目进行配置的文件，包含了项目运行所需的python库和flower相关配置
3. 制作后上传到公开docker镜像平台，开放拉取权限。该镜像不区分server和client，均使用同一镜像

### 5.2 创建、部署server端superexec实例
server端的superexec实例是用于和superlink通信并真正创建任务执行app，由superlink发送任务给superexec，superexec创建对应的serverapp执行server端的任务
```yaml
# 创建配置文件superexec-serverapp-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: superexec-serverapp
  namespace: flower-superlink
spec:
  replicas: 1
  selector:
    matchLabels:
      app: superexec-serverapp
  template:
    metadata:
      labels:
        app: superexec-serverapp
    spec:
      containers:
      - name: superexec-serverapp
        image: ccr.ccs.tencentyun.com/czfflower/flwr_superexec:0.0.1
        args:
          # 连接 SuperLink ServerAppIO API（同 namespace，直接用 ClusterIP Service 名）
          - "--insecure"
          - "--appio-api-address"
          - "superlink-server-svc:9091"
          # 选择 serverapp 插件
          - "--plugin-type"
          - "serverapp"
        ports:
        - containerPort: 9091
        env:
        - name: HF_ENDPOINT
          value: "https://hf-mirror.com"
        - name: HF_HOME
          value: "/app/.cache/huggingface"
        volumeMounts:
        - name: flwr-state
          mountPath: /app/.flwr
        - name: model-cache
          mountPath: /app/.cache
        resources:
          requests:
            cpu: "1"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "16Gi"
      volumes:
      - name: flwr-state
        emptyDir:
          sizeLimit: 200Mi
      - name: model-cache
        emptyDir:
          sizeLimit: 50Gi
```
1. openllama微调使用的模型和数据集都来自huggingface，因此需配置镜像站和登录密钥，由实例自行拉取
2. image填写上一步上传docker镜像的平台链接，需要开放拉取权限
3. args中包含了flower框架下superexec所需的参数，其中appio-api-address是指定与superlink通信的端口的，需要保证和superlink中配置相同，地址则指定为superlink对应地址
4. volume存储方面可以配置持久化，需要注意大小，由于openllama微调模型和数据集都相对较大，需要分配足够的空间

配置service提供9091端口服务，创建文件superlink-server-svc.yaml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: superlink-server-svc
  namespace: flower-superlink
spec:
  selector:
    app: superlink
  ports:  # like a dynamic IP routing table/mapping that routes traffic to the designated ports
  - protocol: TCP
    port: 9091
    targetPort: 9091
  type: ClusterIP
```
该服务用于保证superlink服务的9091端口可以正常与superexec通信，其中name：superlink-server-svc需要与上面superexec实例配置中superlink-server-svc:9091保持一致
启动superexec实例和对应service
```bash
export KUBECONFIG=$KUBECONFIG_CENTER

kubectl apply -f superexec-serverapp-deployment.yaml -n flower-superlink
kubectl apply -f superlink-server-svc.yaml -n flower-superlink

# 确认启动成功后，查看对应log，验证superexec是否和superlink连接成功
kubectl logs -l app=superexec-serverapp -n flower-superlink --tail=20
# 期望输出如下：
# INFO :      Starting Flower SuperExec
# ......
# INFO :      Connection successful after 17.04 seconds and 6 tries.
```

### 5.3 创建、部署client端superexec实例
client端的superexec实例是用于和supernode通信并创建执行训练任务的clientapp的
与server端类似，创建配置文件superexec-a-deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: superexec-clientapp-a
  namespace: flower-supernode-a       # 与 SuperNode B 同 namespace
spec:
  replicas: 1
  selector:
    matchLabels:
      app: superexec-clientapp-a
  template:
    metadata:
      labels:
        app: superexec-clientapp-a
    spec:
      containers:
      - name: superexec-clientapp
        image: ccr.ccs.tencentyun.com/czfflower/flwr_superexec:0.0.1
        args:
          - "--insecure"
          - "--appio-api-address"
          - "supernode-a-svc:9094"
          - "--plugin-type"
          - "clientapp"
        ports:
        - containerPort: 9094
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: HF_ENDPOINT
          value: "https://hf-mirror.com"
        - name: HF_HUB_ENABLE_HF_TRANSFER
          value: "0"
        - name: HF_HOME
          value: "/app/.cache/huggingface"
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: HF_TOKEN
              optional: true
        volumeMounts:
        - name: cache-volume
          mountPath: /app/.cache
        - name: flwr-volume
          mountPath: /app/.flwr
        - name: config-volume
          mountPath: /app/.config
        resources:
          limits:
            nvidia.com/gpu: "1"
            memory: "16Gi"
            cpu: "16"
          requests:
            nvidia.com/gpu: "1"
            memory: "8Gi"
            cpu: "4"
      volumes:
      - name: cache-volume
        emptyDir:
          sizeLimit: 50Gi
      - name: flwr-volume
        emptyDir:
          sizeLimit: 50Mi
      - name: config-volume
        emptyDir:
          sizeLimit: 50Mi
      nodeSelector:
        node.tke.cloud.tencent.com/accelerator-type: gpu
```
1. 开放端口方面，与server端不同，flower官方默认使用9094
2. 由于client端需要使用gpu资源训练，需要指定带有gpu资源的节点并且给出配置（如果server端聚合参数后需要推理验证，则也需要gpu资源）
3. flower参数方面，plugin-type需要改为clientapp

同样为了保证通信，创建service配置文件supernode-a-svc.yaml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: supernode-a-svc
  namespace: flower-supernode-a
spec:
  selector:
    app: supernode-a
  ports:
  - name: clientapp-io
    port: 9094
    targetPort: 9094
  type: ClusterIP
```

启动superexec实例和对应service
```bash
export KUBECONFIG=$KUBECONFIG_TKE

kubectl apply -f superexec-a-deployment.yaml -n flower-supernode-a
kubectl apply -f supernode-a-svc.yaml -n flower-supernode-a

# 确认启动成功后，查看对应log，验证superexec是否和superlink连接成功
kubectl logs -l app=superexec-clientapp-a -n flower-supernode-a --tail=20
# 期望输出如下：
# INFO :      Starting Flower SuperExec
# ......
# INFO :      Connection successful after 17.04 seconds and 6 tries.
```
其他supernode同理，每个supernode都需要对应服务

---


## 阶段六：启动训练任务
确保上述skupper、superlink、supernode、superexec全部正确运行且正确通信后，即可开始任务
1. 下载并安装flwr cli(本地)
```bash
pip install flwr==1.28.0
```
2. 配置flower
创建或找到flwr的config文件（一般在~/.flwr/目录下）
配置如下信息
```yaml
[superlink.cross-cloud]
address = "127.0.0.1:9093"
insecure = true
```
其中cross-cloud表示联邦学习superlink的名字，类似于namespace，可自行取名，不同名字下可配置不同参数，执行任务时指定该名字，则任务按照改名字下配置运行
insecure表示不使用TLS连接，若使用，可配置对应cert等文件地址
address表示flwr cli工具连接地址，如果进行远程操控，并且远程服务器对应端口开放，可以配置远程地址和端口
3. 配置集群端口转发
```bash
export KUBECONFIG=$KUBECONFIG_CENTER

kubectl -n flower-superlink port-forward svc/superlink-service 9093:9093 &
```
通过配置端口转发，可以直接将上面配置中address = "127.0.0.1:9093"等效为在远程集群上部署任务
4. 进入项目根目录（pyproject.toml所在目录），使用flwr cli命令行工具运行下面指令：
```bash
flwr run . cross-cloud

# 期望输出：
# INFO :      Starting logstream for run_id `11981062330485651084`
# INFO :      Start `flwr-serverapp` process
# Successfully installed flowertune-llm to /app/.flwr/apps/flwrlabs.flowertune-llm.1.0.0.edc16e2b.
# WARNING :   fraction_evaluate is set to 0.0. Federated evaluation will be skipped.
# INFO :      Starting FedAvg strategy:
# INFO :          ├── Number of rounds: 100
# INFO :          ├── ArrayRecord (40.64 MB)
# INFO :          ├── ConfigRecord (train): {'save_path': /app/results/2026-05-08_09-07-22}
# INFO :          ├── ConfigRecord (evaluate): (empty!)
# INFO :          ├──> Sampling:
# INFO :          │       ├──Fraction: train (0.10) | evaluate ( 0.00)
# INFO :          │       ├──Minimum nodes: train (2) | evaluate (0)
# INFO :          │       └──Minimum available nodes: 2
# INFO :          └──> Keys in records:
# INFO :                  ├── Weighted by: 'num-examples'
# INFO :                  ├── ArrayRecord key: 'arrays'
# INFO :                  └── ConfigRecord key: 'config'
# INFO :
# INFO :      Initial global evaluation results: {}
```
其中“.”表示要执行的任务所在目录，cross-cloud指定在哪个联邦学习配置下进行任务
初次执行由于需要下载模型、数据集，可能需要等一段时间才会有日志
同时，正确运行后，可观察supernode端superexec服务的日志，期望输出如下：
```bash
# INFO :      Starting Flower SuperExec
# ......
# INFO :      Connection successful after 17.04 seconds and 6 tries.
# INFO :      Start `flwr-clientapp` process
# INFO :      [flwr-clientapp] Pull `AppInputs` for token 4ce8...4f42
# Successfully installed flowertune-llm to /app/.flwr/apps/flwrlabs.flowertune-llm.1.0.0.edc16e2b.
# Generating train split: 100%|██████████| 52002/52002 [00:00<00:00, 230338.08 examples/s]
# ......
# Map: 100%|██████████| 26001/26001 [00:03<00:00, 6671.09 examples/s]
# /python/venv/lib/python3.13/site-packages/trl/trainer/sft_trainer.py:322: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `SFTTrainer.__init__`. Use `processing_class` instead.
#   super().__init__(
# No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
#   0%|          | 0/10 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
# {'loss': 1.4621, 'grad_norm': 0.48439469933509827, 'learning_rate': 4.998791072896043e-05, 'epoch': 0.01}
# {'train_runtime': 120.4731, 'train_samples_per_second': 1.328, 'train_steps_per_second': 0.083, 'train_loss': 1.4621461868286132, 'epoch': 0.01}
# 100%|██████████| 10/10 [02:00<00:00, 12.05s/it]
# INFO :      [flwr-clientapp] Push `AppOutputs` for token 4ce8...4f42
```
指定的全部supernode都完成本轮训练后，server端会下发第下一轮训练任务，日志与上面类似
---

## 其他注意事项

**镜像拉取失败**：若镜像拉取失败，可以换加速源

**superexec连接失败**：若superexec实例启动后，一直显示“INFO :      Starting Flower SuperExec”，可以重启对应superexec和service来解决，若没有解决说明配置有问题

**client端superexec分配空间不足**：若分配空间不足，superexec实例会失败并导致整个联邦学习过程卡住，因为server端的superlink只有收到所有supernode本轮的训练数据后才会执行聚合逻辑并下发下一轮训练任务，只要有一个supernode没返回结果，superlink都会卡住，解决办法是分配足够大的空间后重新启动训练任务
