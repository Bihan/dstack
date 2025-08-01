# dstack Architecture Deep Dive

This document provides a comprehensive overview of dstack's architecture, focusing on the server, shim, and runner components, based on code analysis and architectural exploration.

## Table of Contents

1. [dstack Server Overview](#dstack-server-overview)
2. [Communication Architecture](#communication-architecture)
3. [Shim Component Deep Dive](#shim-component-deep-dive)
4. [Runner Component Deep Dive](#runner-component-deep-dive)
5. [Complete Communication Flow](#complete-communication-flow)
6. [Port Management and Networking](#port-management-and-networking)
7. [GPU Management](#gpu-management)
8. [Example: Dev Environment Lifecycle](#example-dev-environment-lifecycle)

## dstack Server Overview

### What is `dstack server`?

The `dstack server` command starts the **central orchestration component** of the dstack platform - a FastAPI-based web server using Uvicorn.

### Key Server Functions

#### **Primary Components**
- **FastAPI Web Server**: Serves REST API endpoints and web UI
- **Database Management**: Handles projects, users, runs, and metadata
- **Background Processing**: Manages job scheduling and resource lifecycle
- **SSH Tunnel Management**: Establishes secure connections to remote resources

#### **Server Initialization Process**

When you run `dstack server`:

1. **Startup Sequence**:
   ```bash
   $ dstack server
   # Server binds to 127.0.0.1:3000 (default)
   # Displays dstack ASCII logo
   # Configures logging and database
   # Creates admin user and generates token
   # Sets up default project configuration
   # Starts background task scheduler
   ```

2. **Output Example**:
   ```
   The admin token is "bbae0f28-d3dd-4820-bf61-8f4bb40815da"
   The dstack server is running at http://127.0.0.1:3000/
   ```

#### **API Endpoints**

The server provides REST API endpoints for:
- **Projects**: Project management and configuration  
- **Users**: User authentication and management
- **Backends**: Cloud provider configurations (AWS, GCP, Azure, etc.)
- **Fleets**: Cluster management for on-premises and cloud resources
- **Runs**: Job execution and lifecycle management
- **Instances**: Compute resource provisioning and management
- **Volumes**: Persistent storage management
- **Gateways**: Service publishing with custom domains and HTTPS
- **Secrets**: Secure credential storage
- **Logs**: Job output and monitoring
- **Metrics**: Performance and resource usage data

#### **Background Tasks**

The server runs scheduled background tasks for:
- Processing submitted jobs and runs
- Managing running and terminating jobs  
- Handling instance lifecycle (provisioning, monitoring, cleanup)
- Processing fleet operations
- Volume management and cleanup
- Metrics collection
- Gateway management

### Host and Port Binding

**"Binding to a host and port"** refers to **network binding** - where the server listens for connections:

- **Host (127.0.0.1)**: The "localhost" or "loopback" address
  - Only accepts connections from the same machine
  - Other processes on your computer can connect, but external machines cannot
  - Security measure - server only accessible locally by default

- **Port (3000)**: The specific port number where the server listens
  - Only one application can bind to a specific port at a time
  - Think of it like an apartment number - host is building address, port is unit number

**Custom Configuration Examples**:
```bash
# Default (local-only access)
$ dstack server
# Accessible at: http://127.0.0.1:3000

# Network access (all interfaces)
$ dstack server --host 0.0.0.0 --port 8080
# Accessible at: http://your-machine-ip:8080

# Specific interface
$ dstack server --host 192.168.1.100 --port 3000
# Only accessible via that specific IP
```

## Communication Architecture

### Overview

dstack uses a **dual SSH tunnel architecture** for secure communication between components:

```
Your Terminal ──HTTP──> dstack-server ──SSH tunnels──> Remote VM
     (CLI)           (localhost:3000)                    (shim + containers)
```

### Complete Communication Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Your Terminal  │    │  dstack-server   │    │    VM Host      │    │   Container      │
│                 │    │  (localhost)     │    │                 │    │                  │
│ dstack apply    ├────┤ Port: 3000       ├────┤ shim-web-server ├────┤ runner-web-server│
│ dstack ps       │HTTP│                  │SSH │ (task mgmt)     │SSH │ (job execution)  │
│ dstack logs     │    │                  │    │                 │    │                  │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
                              │                        │                        │
                              │                   Docker API                Your Code
                              │                        │                        │
                              └────────── Orchestrates ─────────────────────────┘
```

### Two Independent Communication Channels

#### **Channel 1: Server ↔ Shim**
- **Purpose**: Infrastructure management (containers, GPUs, volumes)
- **Transport**: SSH tunnel from server to VM host
- **Protocol**: HTTP over SSH tunnel
- **Endpoints**: 
  - `POST /tasks` - Submit container tasks
  - `GET /tasks/{id}` - Get task status  
  - `DELETE /tasks/{id}` - Terminate tasks
  - `GET /resources` - Get available resources

#### **Channel 2: Server ↔ Runner**  
- **Purpose**: Job execution (code, logs, status)
- **Transport**: SSH tunnel from server to container
- **Protocol**: HTTP over SSH tunnel
- **Endpoints**:
  - Job specification submission
  - Code upload (git/tarball)
  - Log streaming via HTTP
  - Status reporting

### SSH Tunnel Details

**Authentication & Security**:
- **Key-based authentication**: Project SSH keys uploaded during provisioning
- **No password authentication**: `PasswordAuthentication=no`
- **Keep-alive settings**: 
  - `ClientAliveInterval=30` (health checks every 30 seconds)
  - `ClientAliveCountMax=4` (timeout after 4 failed checks)

## Shim Component Deep Dive

### What is the Shim?

The **dstack-shim** is a **container orchestrator** that runs on VM hosts and manages Docker containers on behalf of the dstack server.

### Shim Architecture

```go
type DockerRunner struct {
    client       *docker.Client      // Docker API client
    dockerParams DockerParameters    // Configuration parameters
    dockerInfo   dockersystem.Info   // Docker daemon info
    gpus         []host.GpuInfo      // Available GPUs on host
    gpuVendor    common.GpuVendor    // GPU vendor (NVIDIA, AMD, etc.)
    gpuLock      *GpuLock           // GPU resource locking
    tasks        TaskStorage         // In-memory task storage
}
```

### Shim Initialization Process

1. **Connect to Docker daemon**:
   ```go
   client, err := docker.NewClientWithOpts(docker.FromEnv, docker.WithAPIVersionNegotiation())
   ```

2. **Detect GPU hardware**:
   ```go
   gpus := host.GetGpuInfo(ctx)
   if len(gpus) > 0 {
       gpuVendor = gpus[0].Vendor
   }
   ```

3. **Initialize GPU resource locking**:
   ```go
   gpuLock, err := NewGpuLock(gpus)
   ```

4. **Restore state from existing containers** (crash recovery):
   ```go
   if err := runner.restoreStateFromContainers(ctx); err != nil {
       return nil, tracerr.Errorf("failed to restore state from containers: %w", err)
   }
   ```

### Task Lifecycle Management

#### **Task Submission (`Submit`)**
- Creates new task from configuration
- Adds to in-memory task storage
- Task starts in `TaskStatusPending` state

#### **Task Execution (`Run`)**
Complete execution flow:

```go
func (d *DockerRunner) Run(ctx context.Context, taskID string) error {
    // 1. PREPARING PHASE
    task.SetStatusPreparing()
    
    // 2. CREATE RUNNER DIRECTORY
    runnerDir, err := d.dockerParams.MakeRunnerDir(task.containerName)
    
    // 3. GPU RESOURCE ALLOCATION
    if cfg.GPU != 0 {
        gpuIDs, err := d.gpuLock.Acquire(ctx, cfg.GPU)
        task.gpuIDs = gpuIDs
    }
    
    // 4. SSH KEY SETUP
    if len(cfg.HostSshKeys) > 0 {
        ak := AuthorizedKeys{user: cfg.HostSshUser}
        ak.AppendPublicKeys(cfg.HostSshKeys)
    }
    
    // 5. VOLUME PREPARATION
    prepareVolumes(ctx, cfg)
    prepareInstanceMountPoints(cfg)
    
    // 6. IMAGE PULLING PHASE
    task.SetStatusPulling(cancelPull)
    pullImage(pullCtx, d.client, cfg, pullLogPath)
    
    // 7. CONTAINER CREATION PHASE
    task.SetStatusCreating()
    d.createContainer(ctx, &task)
    
    // 8. CONTAINER EXECUTION PHASE
    task.SetStatusRunning()
    d.startContainer(ctx, &task)
    d.waitContainer(ctx, &task)
}
```

### GPU Management in Shim

#### **GPU Discovery and Locking**
The shim manages GPU resources across multiple tasks:

```go
// GPU allocation during PREPARING phase
if cfg.GPU != 0 {
    gpuIDs, err := d.gpuLock.Acquire(ctx, cfg.GPU)
    if err != nil {
        task.SetStatusTerminated(string(types.TerminationReasonExecutorError), err.Error())
        return tracerr.Wrap(err)
    }
    task.gpuIDs = gpuIDs
    
    defer func() {
        releasedGpuIDs := d.gpuLock.Release(ctx, task.gpuIDs)
        log.Debug(ctx, "released GPU(s)", "task", task.ID, "gpus", releasedGpuIDs)
    }()
}
```

#### **Multi-Vendor GPU Support**

**NVIDIA GPUs**:
```go
case common.GpuVendorNvidia:
    hostConfig.Resources.DeviceRequests = append(
        hostConfig.Resources.DeviceRequests,
        container.DeviceRequest{
            Capabilities: [][]string{{"gpu", "utility", "compute", "graphics", "video", "display", "compat32"}},
            DeviceIDs:    ids,
        },
    )
```

**AMD GPUs**:
```go
case common.GpuVendorAmd:
    // Mount KFD device
    hostConfig.Resources.Devices = append(hostConfig.Resources.Devices,
        container.DeviceMapping{
            PathOnHost: "/dev/kfd",
            PathInContainer: "/dev/kfd",
            CgroupPermissions: "rwm",
        },
    )
    // Mount render nodes
    for _, renderNodePath := range ids {
        hostConfig.Resources.Devices = append(hostConfig.Resources.Devices,
            container.DeviceMapping{
                PathOnHost: renderNodePath,
                PathInContainer: renderNodePath,
                CgroupPermissions: "rwm",
            },
        )
    }
```

### Image Pulling Process

The shim is responsible for pulling Docker images:

```go
func pullImage(ctx context.Context, client docker.APIClient, taskConfig TaskConfig, logPath string) error {
    // Add :latest tag if no tag specified
    if !strings.Contains(taskConfig.ImageName, ":") {
        taskConfig.ImageName += ":latest"
    }
    
    // Check if image already exists locally
    images, err := client.ImageList(ctx, image.ListOptions{
        Filters: filters.NewArgs(filters.Arg("reference", taskConfig.ImageName)),
    })
    
    // Skip pull if image exists (unless it's :latest)
    if len(images) > 0 && !strings.Contains(taskConfig.ImageName, ":latest") {
        return nil
    }
    
    // Setup registry authentication if provided
    opts := image.PullOptions{}
    regAuth, err := encodeRegistryAuth(taskConfig.RegistryUsername, taskConfig.RegistryPassword)
    if regAuth != "" {
        opts.RegistryAuth = regAuth
    }
    
    // Pull the image with timeout protection
    reader, err := client.ImagePull(ctx, taskConfig.ImageName, opts)
    // ... progress tracking and logging
}
```

**Key Features**:
- **Timeout Protection**: 20-minute timeout to prevent hanging
- **Progress Logging**: Saves pull progress to `pull.log` file
- **Registry Authentication**: Supports private registries
- **Smart Caching**: Skips pull if image already exists (except `:latest`)

### Container Setup and SSH Configuration

The shim configures containers with comprehensive SSH setup:

```go
func getSSHShellCommands(openSSHPort int, publicSSHKey string) []string {
    return []string{
        // Package manager detection
        `if _exists apt-get; then _install() { apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "$1"; }; fi`,
        
        // Install SSH server if not present
        `if ! _exists sshd; then _install openssh-server; fi`,
        
        // Configure SSH
        "mkdir -p ~/.ssh",
        "chmod 700 ~/.ssh",
        fmt.Sprintf("echo '%s' > ~/.ssh/authorized_keys", publicSSHKey),
        
        // Start SSH daemon
        fmt.Sprintf("/usr/sbin/sshd -p %d -o PidFile=none -o PasswordAuthentication=no", openSSHPort),
        
        // Start dstack-runner
        fmt.Sprintf("%s %s", consts.RunnerBinaryPath, strings.Join(runnerArgs, " ")),
    }
}
```

### State Recovery

The shim can recover from crashes by inspecting existing containers:

```go
func (d *DockerRunner) restoreStateFromContainers(ctx context.Context) error {
    // Find containers with dstack labels
    listOptions := container.ListOptions{
        All: true,
        Filters: filters.NewArgs(filters.Arg("label", fmt.Sprintf("%s=%s", LabelKeyIsTask, LabelValueTrue))),
    }
    
    containers, err := d.client.ContainerList(ctx, listOptions)
    
    // Restore task state and GPU locks
    for _, containerShort := range containers {
        taskID := containerShort.Labels[LabelKeyTaskID]
        // Recreate task objects and GPU locks
        task := NewTask(taskID, status, containerName, containerID, gpuIDs, ports, runnerDir)
        d.tasks.Add(task)
    }
}
```

## Runner Component Deep Dive

### What is the Runner?

The **dstack-runner** is the component that runs **inside Docker containers** and executes actual user workloads. It serves as the job executor while the shim handles infrastructure.

### Runner Architecture

The runner follows a **linear lifecycle**:

1. **STEP 1: Wait for job spec**
   - Receives job specification from dstack server
   - Gets environment variables, secrets, commands to run

2. **STEP 2: Wait for code**
   - Receives either:
     - **Tarball**: Complete code archive
     - **Diff**: Git repository changes

3. **STEP 3: Prepare repository**
   - **Option A**: Clone git repo and apply diff
   - **Option B**: Extract tarball archive
   - Sets up the working directory

4. **STEP 4: Execute commands**
   - Runs user-specified commands from job spec
   - **Serves logs** to dstack server via HTTP API
   - **Streams real-time logs** to CLI via WebSocket
   - **Waits for termination signal** from server

5. **STEP 5: Cleanup**
   - Ensures all logs are read by server and CLI
   - Exits after timeout if logs aren't fully consumed

### Runner Responsibilities

- **Environment setup**: Sets environment variables and secrets
- **Command execution**: Runs user's Python scripts, training jobs, etc.
- **Log management**: Collects and serves both job logs and runner logs
- **Status reporting**: Reports job status back to dstack server
- **Signal handling**: Gracefully terminates on server request

### Runner vs Shim Separation

| Aspect | Shim | Runner |
|--------|------|--------|
| **Location** | VM host | Inside container |
| **Purpose** | Infrastructure management | Job execution |
| **Privileges** | Host privileges (Docker, GPU) | Container privileges |
| **Communication** | Synchronous API calls | Long-running job execution |
| **Responsibilities** | Container lifecycle, GPU allocation | Code execution, log streaming |

## Port Management and Networking

### Port Architecture Overview

Each container requires multiple ports for different services:

```go
func (c *CLIArgs) DockerPorts() []int {
    return []int{c.Runner.HTTPPort, c.Runner.SSHPort}
}
```

### Container Port Configuration

#### **Fixed Internal Ports** (inside container):
- **Runner SSH**: `c.Runner.SSHPort` (e.g., 10022) - **FIXED**
- **Runner HTTP**: `c.Runner.HTTPPort` (e.g., 10999) - **FIXED**

#### **Ephemeral Host Ports** (on host):
```go
func bindPorts(ports []int) nat.PortMap {
    portMap := make(nat.PortMap)
    for _, port := range ports {
        portMap[nat.Port(fmt.Sprintf("%d/tcp", port))] = []nat.PortBinding{
            {
                HostIP:   "0.0.0.0",
                HostPort: "", // ← Empty string = use ephemeral port
            },
        }
    }
    return portMap
}
```

### Port Mapping Example

```
Inside Container (FIXED ports):
┌─────────────────────────────┐
│ SSH Server: 10022           │
│ Runner HTTP API: 10999      │
└─────────────────────────────┘
                │
                │ Docker port mapping
                ▼
VM Host (EPHEMERAL ports):
┌─────────────────────────────┐  
│ Host Port: 45123 → 10022    │
│ Host Port: 45124 → 10999    │
└─────────────────────────────┘
```

### Network Modes

#### **Bridge Mode** (default):
```
dstack-server ──SSH tunnel──> Host:45123 ──maps to──> Container:10022 (SSH)
dstack-server ──SSH tunnel──> Host:45124 ──maps to──> Container:10999 (HTTP)
```

#### **Host Mode**:
```go
if getNetworkMode(task.config.NetworkMode).IsHost() {
    task.ports = []PortMapping{}  // No port mapping needed
    return nil
}
```

In host mode:
```
dstack-server ──SSH tunnel──> Host:10022 (directly to container)
dstack-server ──SSH tunnel──> Host:10999 (directly to container)
```

### Why Ephemeral Ports?

1. **Avoid Conflicts**: Multiple containers can't bind to same host port
2. **Automatic Management**: No need to manually track which ports are free  
3. **Scalability**: Can run many containers without port planning
4. **Security**: Harder to predict which ports services are running on

## Complete Communication Flow

### Example: `dstack apply -f config.yml`

#### **Phase 1: Infrastructure Setup**
```
1. CLI ──HTTP──> dstack-server:3000
   Request: "Create dev environment with 1 GPU"

2. dstack-server ──SSH tunnel──> VM-Host:22 ──HTTP──> shim-web-server
   Request: POST /tasks {"image": "dstack:py3.12-cuda", "gpu": 1}

3. Shim creates container:
   - Pulls image: dstack:py3.12-cuda
   - Allocates GPU: GPU-12345678-1234-1234-1234-123456789012
   - Port mapping: 10022→45123, 10999→45124 (ephemeral host ports)
   - Starts container with SSH server + runner
```

#### **Phase 2: Job Execution**
```
4. dstack-server ──SSH tunnel──> VM-Host:45123 ──forwards to──> Container:10022
   Establishes SSH connection to runner inside container

5. dstack-server ──HTTP over SSH──> runner-web-server:10999
   Request: "Execute Python training script"

6. Runner inside container:
   - Downloads your code (git repo/tarball)
   - Sets environment variables
   - Executes: python train.py
   - Streams logs back via HTTP API
```

#### **Phase 3: Real-time Communication**
```
7. Container ──logs──> runner-HTTP-API ──SSH tunnel──> dstack-server ──WebSocket──> Your CLI
   Real-time log streaming: "Epoch 1/10: loss=0.5..."

8. Your CLI ──HTTP──> dstack-server ──SSH tunnel──> runner-HTTP-API
   Status checks, termination requests, etc.
```

### Communication Summary

- **Your Local Port**: `3000` (dstack server)
- **Shim Communication**: Via SSH tunnel + HTTP API (infrastructure)
- **Runner Communication**: Via separate SSH tunnel + HTTP API (jobs)
- **All Remote Communication**: Encrypted via SSH tunnels
- **No Direct Access**: You only interact with local port 3000

## GPU Management

### GPU Discovery and Allocation

During the **PREPARING** stage, the shim allocates GPUs:

1. **Discovery at startup**:
   ```go
   gpus := host.GetGpuInfo(ctx)
   if len(gpus) > 0 {
       gpuVendor = gpus[0].Vendor
   }
   gpuLock, err := NewGpuLock(gpus)
   ```

2. **Allocation during task preparation**:
   ```go
   if cfg.GPU != 0 {
       gpuIDs, err := d.gpuLock.Acquire(ctx, cfg.GPU)
       task.gpuIDs = gpuIDs
       log.Debug(ctx, "acquired GPU(s)", "task", task.ID, "gpus", gpuIDs)
   }
   ```

3. **Vendor-specific configuration**:
   - **NVIDIA**: Uses GPU UUIDs like `"GPU-12345678-1234-1234-1234-123456789012"`
   - **AMD**: Uses render node paths like `"/dev/dri/renderD128"`
   - **Tenstorrent**: Uses device indices like `"0", "1"`
   - **Intel**: Uses device indices for Habana devices

### GPU Lock Management

- **Exclusive Access**: Each GPU can only be used by one task at a time
- **Automatic Cleanup**: GPUs released automatically when task terminates
- **Crash Recovery**: GPU locks restored by inspecting existing containers
- **Error Handling**: Task terminated if GPU allocation fails

## Example: Dev Environment Lifecycle

### Configuration Application Process

When you run `dstack apply -f config.yml` with:
```yaml
type: dev-environment
name: cursor
python: 3.12
ide: cursor
files: 
  - .:examples
  - ~/.ssh/id_rsa:/root/.ssh/id_rsa
resources:
  gpu: 1
```

### Step-by-Step Process

1. **Configuration Loading**: CLI loads and parses YAML configuration
2. **Backend Selection**: Uses specified backend (e.g., `-b gcp`)
3. **Offer Discovery**: Backend queries cloud provider for GPU instances
4. **User Confirmation**: Presents available offers with pricing
5. **Infrastructure Provisioning**: 
   - Creates VM instance
   - Configures GPU drivers
   - Sets up SSH access
6. **Environment Setup**:
   - Uploads environment variables
   - Installs dstack components
   - Sets up IDE (Cursor) with remote access
7. **File Mounting**:
   - Maps local directory to container
   - Mounts SSH keys
   - Uploads repo changes
8. **IDE Access**: Provides remote connection URL

### Key Components Involved

- **CLI**: `ApplyCommand` orchestrates the process
- **Backend**: Cloud-specific compute handlers
- **Configurators**: Process configuration requirements
- **Server**: Background tasks handle deployment
- **Shim**: Manages container lifecycle on VM
- **Runner**: Executes workloads inside container

## Key Architectural Benefits

1. **Security**: All communication encrypted via SSH tunnels
2. **Scalability**: Multiple containers managed per host
3. **Isolation**: Separate components for infrastructure vs. application logic
4. **Fault Tolerance**: Crash recovery and state restoration
5. **Flexibility**: Works across different cloud providers
6. **Resource Management**: Intelligent GPU allocation and cleanup
7. **Separation of Concerns**: Clear boundaries between components

This architecture provides a robust, secure, and scalable platform for managing AI workloads across diverse cloud and on-premises infrastructure.
