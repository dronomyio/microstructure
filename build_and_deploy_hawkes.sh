#!/bin/bash

# Ultra-Fast Hawkes Process Engine - Build and Deployment Script
# ==============================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="ultra-fast-hawkes-engine"
BUILD_TYPE="${BUILD_TYPE:-Release}"
DOCKER_TAG="${DOCKER_TAG:-latest}"
PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc)}"
ENABLE_TESTS="${ENABLE_TESTS:-true}"
ENABLE_BENCHMARKS="${ENABLE_BENCHMARKS:-true}"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check for required tools
    local missing_tools=()
    
    if ! command -v cmake &> /dev/null; then
        missing_tools+=("cmake")
    fi
    
    if ! command -v nvcc &> /dev/null; then
        missing_tools+=("nvcc (CUDA toolkit)")
    fi
    
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        missing_tools+=("docker-compose")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        print_error "Please install the missing tools and try again."
        exit 1
    fi
    
    # Check CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        print_status "CUDA version: $CUDA_VERSION"
        
        if [[ $(echo "$CUDA_VERSION < 11.0" | bc -l) -eq 1 ]]; then
            print_warning "CUDA version $CUDA_VERSION is older than recommended (11.0+)"
        fi
    fi
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_status "Found $GPU_COUNT GPU(s)"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read line; do
            print_status "  GPU: $line"
        done
    else
        print_warning "nvidia-smi not found. GPU support may not be available."
    fi
    
    # Check CPU SIMD capabilities
    print_status "CPU SIMD capabilities:"
    if grep -q avx512 /proc/cpuinfo; then
        print_status "  ✓ AVX-512 supported"
    elif grep -q avx2 /proc/cpuinfo; then
        print_status "  ✓ AVX-256 supported"
    else
        print_status "  ⚠ Basic SIMD only"
    fi
    
    print_success "Prerequisites check completed"
}

# Function to clean build directory
clean_build() {
    print_status "Cleaning build directory..."
    
    if [ -d "build" ]; then
        rm -rf build
        print_status "Removed existing build directory"
    fi
    
    mkdir -p build
    print_success "Created clean build directory"
}

# Function to configure CMake
configure_cmake() {
    print_status "Configuring CMake..."
    
    cd build
    
    # CMake configuration options
    CMAKE_OPTS=(
        "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        "-DCMAKE_INSTALL_PREFIX=/usr/local"
    )
    
    # Add CUDA architectures if specified
    if [ -n "$CUDA_ARCHITECTURES" ]; then
        CMAKE_OPTS+=("-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES")
    fi
    
    # Add additional options based on build type
    case "$BUILD_TYPE" in
        "Debug")
            CMAKE_OPTS+=("-DCMAKE_CUDA_FLAGS=-g -G -O0")
            CMAKE_OPTS+=("-DCMAKE_CXX_FLAGS=-g -O0 -Wall -Wextra")
            ;;
        "Release")
            CMAKE_OPTS+=("-DCMAKE_CUDA_FLAGS=-O3 -DNDEBUG --use_fast_math")
            CMAKE_OPTS+=("-DCMAKE_CXX_FLAGS=-O3 -DNDEBUG -march=native -mtune=native")
            ;;
        "RelWithDebInfo")
            CMAKE_OPTS+=("-DCMAKE_CUDA_FLAGS=-O2 -g -DNDEBUG")
            CMAKE_OPTS+=("-DCMAKE_CXX_FLAGS=-O2 -g -DNDEBUG")
            ;;
    esac
    
    # Run CMake
    #cmake "${CMAKE_OPTS[@]}" -f ../CMakeLists_hawkes.txt ..
    cmake "${CMAKE_OPTS[@]}" ..
    
    cd ..
    print_success "CMake configuration completed"
}

# Function to build the project
build_project() {
    print_status "Building project with $PARALLEL_JOBS parallel jobs..."
    
    cd build
    
    # Build the project
    make -j$PARALLEL_JOBS
    
    cd ..
    print_success "Project build completed"
}

# Function to run tests
run_tests() {
    if [ "$ENABLE_TESTS" = "true" ]; then
        print_status "Running tests..."
        
        cd build
        
        if [ -f "hawkes_test" ]; then
            ./hawkes_test
            print_success "All tests passed"
        else
            print_warning "Test executable not found"
        fi
        
        cd ..
    else
        print_status "Tests disabled"
    fi
}

# Function to run benchmarks
run_benchmarks() {
    if [ "$ENABLE_BENCHMARKS" = "true" ]; then
        print_status "Running benchmarks..."
        
        cd build
        
        if [ -f "hawkes_benchmark" ]; then
            ./hawkes_benchmark
            print_success "Benchmarks completed"
        else
            print_warning "Benchmark executable not found"
        fi
        
        cd ..
    else
        print_status "Benchmarks disabled"
    fi
}

# Function to build Docker image
build_docker() {
    print_status "Building Docker image..."
    
    # Build the Docker image
    docker build -f Dockerfile_hawkes -t $PROJECT_NAME:$DOCKER_TAG .
    
    # Tag additional versions
    docker tag $PROJECT_NAME:$DOCKER_TAG $PROJECT_NAME:latest
    
    print_success "Docker image built successfully"
    print_status "Image tags: $PROJECT_NAME:$DOCKER_TAG, $PROJECT_NAME:latest"
}

# Function to test Docker image
test_docker() {
    print_status "Testing Docker image..."
    
    # Run basic functionality test
    docker run --rm --gpus all $PROJECT_NAME:$DOCKER_TAG hawkes_test
    
    print_success "Docker image test completed"
}

# Function to deploy with Docker Compose
deploy_docker_compose() {
    print_status "Deploying with Docker Compose..."
    
    # Create necessary directories
    mkdir -p data output logs
    
    # Create configuration files if they don't exist
    create_config_files
    
    # Start the services
    docker-compose -f docker-compose_hawkes.yml up -d
    
    print_success "Docker Compose deployment completed"
    print_status "Services started. Check status with: docker-compose -f docker-compose_hawkes.yml ps"
}

# Function to create configuration files
create_config_files() {
    print_status "Creating configuration files..."
    
    # Create monitoring directory and Prometheus config
    mkdir -p monitoring
    if [ ! -f "monitoring/prometheus.yml" ]; then
        cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'hawkes-engine'
    static_configs:
      - targets: ['hawkes-engine:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s
EOF
        print_status "Created monitoring/prometheus.yml"
    fi
    
    # Create nginx directory and config
    mkdir -p nginx
    if [ ! -f "nginx/nginx.conf" ]; then
        cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream hawkes_backend {
        least_conn;
        server hawkes-engine:8080;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://hawkes_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }
        
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
EOF
        print_status "Created nginx/nginx.conf"
    fi
    
    # Create file server config
    if [ ! -f "nginx/file-server.conf" ]; then
        cat > nginx/file-server.conf << 'EOF'
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    
    location / {
        autoindex on;
        autoindex_exact_size off;
        autoindex_localtime on;
    }
    
    location /output/ {
        alias /usr/share/nginx/html/output/;
        autoindex on;
    }
    
    location /data/ {
        alias /usr/share/nginx/html/data/;
        autoindex on;
    }
}
EOF
        print_status "Created nginx/file-server.conf"
    fi
    
    # Create SQL init script
    mkdir -p sql
    if [ ! -f "sql/init.sql" ]; then
        cat > sql/init.sql << 'EOF'
CREATE TABLE IF NOT EXISTS hawkes_analysis_runs (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    n_events INTEGER NOT NULL,
    mu REAL NOT NULL,
    alpha REAL NOT NULL,
    beta REAL NOT NULL,
    log_likelihood REAL NOT NULL,
    processing_time_ms REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_hawkes_ticker_time ON hawkes_analysis_runs(ticker, start_time);
EOF
        print_status "Created sql/init.sql"
    fi
    
    print_success "Configuration files created"
}

# Function to show usage
show_usage() {
    echo "Ultra-Fast Hawkes Process Engine - Build and Deployment Script"
    echo "=============================================================="
    echo ""
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       - Build the C++/CUDA project (default)"
    echo "  docker      - Build Docker image"
    echo "  deploy      - Deploy with Docker Compose"
    echo "  test        - Run tests only"
    echo "  benchmark   - Run benchmarks only"
    echo "  clean       - Clean build directory"
    echo "  all         - Build, test, and create Docker image"
    echo "  help        - Show this help message"
    echo ""
    echo "Options:"
    echo "  --build-type TYPE     Build type: Debug, Release, RelWithDebInfo (default: Release)"
    echo "  --docker-tag TAG      Docker image tag (default: latest)"
    echo "  --parallel-jobs N     Number of parallel build jobs (default: $(nproc))"
    echo "  --no-tests           Disable tests"
    echo "  --no-benchmarks      Disable benchmarks"
    echo "  --cuda-arch ARCH     CUDA architectures (e.g., '70;75;80')"
    echo ""
    echo "Environment Variables:"
    echo "  BUILD_TYPE           Build type (Debug, Release, RelWithDebInfo)"
    echo "  DOCKER_TAG           Docker image tag"
    echo "  PARALLEL_JOBS        Number of parallel build jobs"
    echo "  ENABLE_TESTS         Enable/disable tests (true/false)"
    echo "  ENABLE_BENCHMARKS    Enable/disable benchmarks (true/false)"
    echo "  CUDA_ARCHITECTURES   CUDA architectures to build for"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Build with default settings"
    echo "  $0 --build-type Debug build          # Debug build"
    echo "  $0 --no-tests docker                 # Build Docker image without tests"
    echo "  $0 --cuda-arch '70;75;80' all        # Build for specific GPU architectures"
    echo "  $0 deploy                             # Deploy with Docker Compose"
    echo ""
}

# Function to show system information
show_system_info() {
    print_status "System Information:"
    echo "  OS: $(uname -s) $(uname -r)"
    echo "  Architecture: $(uname -m)"
    echo "  CPU cores: $(nproc)"
    echo "  Memory: $(free -h | awk '/^Mem:/ {print $2}')"
    
    if command -v nvidia-smi &> /dev/null; then
        echo "  GPUs: $(nvidia-smi --list-gpus | wc -l)"
    fi
    
    if command -v nvcc &> /dev/null; then
        echo "  CUDA: $(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')"
    fi
    
    echo "  Docker: $(docker --version 2>/dev/null || echo 'Not installed')"
    echo "  Docker Compose: $(docker-compose --version 2>/dev/null || echo 'Not installed')"
    echo ""
}

# Main execution function
main() {
    local command="build"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build-type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            --docker-tag)
                DOCKER_TAG="$2"
                shift 2
                ;;
            --parallel-jobs)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            --cuda-arch)
                CUDA_ARCHITECTURES="$2"
                shift 2
                ;;
            --no-tests)
                ENABLE_TESTS="false"
                shift
                ;;
            --no-benchmarks)
                ENABLE_BENCHMARKS="false"
                shift
                ;;
            build|docker|deploy|test|benchmark|clean|all|help)
                command="$1"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Show header
    echo "=============================================================="
    echo "Ultra-Fast Hawkes Process Engine - Build and Deployment"
    echo "=============================================================="
    echo ""
    
    # Show system information
    show_system_info
    
    # Execute command
    case "$command" in
        "help")
            show_usage
            ;;
        "clean")
            clean_build
            ;;
        "build")
            check_prerequisites
            clean_build
            configure_cmake
            build_project
            run_tests
            run_benchmarks
            print_success "Build completed successfully!"
            ;;
        "docker")
            build_docker
            test_docker
            print_success "Docker build completed successfully!"
            ;;
        "deploy")
            deploy_docker_compose
            print_success "Deployment completed successfully!"
            print_status "Access the application at: http://localhost:8080"
            print_status "Monitor with Grafana at: http://localhost:3000 (admin/hawkes_admin)"
            ;;
        "test")
            cd build 2>/dev/null || { print_error "Build directory not found. Run 'build' first."; exit 1; }
            run_tests
            cd ..
            ;;
        "benchmark")
            cd build 2>/dev/null || { print_error "Build directory not found. Run 'build' first."; exit 1; }
            run_benchmarks
            cd ..
            ;;
        "all")
            check_prerequisites
            clean_build
            configure_cmake
            build_project
            run_tests
            run_benchmarks
            build_docker
            test_docker
            print_success "Complete build and Docker creation finished!"
            ;;
        *)
            print_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Trap to handle interruption
trap 'print_error "Build interrupted by user"; exit 130' INT

# Run main function
main "$@"

