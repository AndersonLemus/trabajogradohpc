# Trabajo de grado.

## Instalación MKL 

Agregar repositorio de Intel.
```
sudo tee /etc/yum.repos.d/oneAPI.repo > /dev/null << \EOF
[oneAPI]
name=Intel® oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF

```
## Instalar la libreria MKL.
```
sudo dnf install intel-oneapi-mkl-devel
```

## SRUN
```
srun -p gpu_titan -w felix --nodes=1 --ntasks=1 --cpus-per-task=8 --time=02:00:00 --pty bash -i
srun -p fat -w thor --nodes=1 --ntasks=1 --cpus-per-task=8 --time=02:00:00 --pty bash -i
srun -p amd -w exadell --nodes=1 --ntasks=1 --exclusive --cpus-per-task=8 --time=48:00:00 --pty bash -i
srun -p amd -w smexa --nodes=1 --ntasks=1 --cpus-per-task=8 --time=02:00:00 --pty bash -i
srun -p GPU -w paccaA100 --nodes=1 --ntasks=1 --cpus-per-task=8 --exclusive --time=48:00:00 --pty bash -i


srun -p gpu_titan -w felix --nodes=1 --ntasks=1 --gres=gpu:1 --time=02:00:00 --pty bash -i
srun -p fat -w thor --nodes=1 --ntasks=1 --gres=gpu:1 --time=02:00:00 --pty bash -i
srun -p amd -w exadell --nodes=1 --ntasks=1 --gres=gpu:1 --mem=32G --time=02:00:00 --pty bash -i
srun -p amd -w smexa --nodes=1 --ntasks=1 --gres=gpu:1 --mem=32G  --time=02:00:00 --pty bash -i
```

## Ejecutar el .sh según corresponda
```
source ./setup_env.sh (Intel)
source ./setup_env_cuda.sh (Nvidia)
source ./setup_env_amd_cpu.sh (AMD)
```

## Compilación

### Thor 
```
g++ -O3 -march=native -std=c++17 src/benchmark_cpu_intel.cpp -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -fopenmp -lpthread -lm -ldl -o bin/cpu_intel_dataset_16

nvcc -O3 -arch=sm_52 -allow-unsupported-compiler benchmark_gpu_nvidia.cu -L/usr/lib64 -lnvidia-ml -lcublas -lcusparse -lcufft -lpthread -o benchmark_titanx
```

### Exadell
```
g++ -O3 -march=znver4 -fopenmp -std=c++17 src/benchmark_cpu_amd.cpp -I /home/jsgalvisb/proyecto/libs/OpenBLAS -I /home/jsgalvisb/proyecto/libs/eigen-3.4.0 -I /home/jsgalvisb/proyecto/libs/fftw-3.3.10/api -L /home/jsgalvisb/proyecto/libs/OpenBLAS -L /home/jsgalvisb/proyecto/libs/fftw-3.3.10/.libs -lopenblas -lfftw3_threads -lfftw3 -lm -lpthread -o bin/benchmark_epyc

hipcc -O3 --offload-arch=gfx90a -std=c++17 benchmark_amd.hip.cpp -lrocblas -lrocsparse -lrocfft -lpthread -o benchmark_exadell_gpu
```

### PACCA
```
g++ -O3 -march=native -std=c++17 src/benchmark_cpu_intel.cpp -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -o bin/cpu_intel_dataset

nvcc -O3 -arch=sm_52 -std=c++11  src/benchmark_gpu_nvidia.cu -L/usr/lib64 -lnvidia-ml -lcublas -lcusparse -lcufft -lpthread -o bin/benchmark_titanx_2
```

Pacca corregido no tocar por ahora
```g++ -O3 -march=native -std=c++17 \
    src/benchmark_cpu_intel.cpp \
    -I${MKLROOT}/include \
    -L${MKLROOT}/lib/intel64 \
    -lmkl_intel_lp64 \
    -lmkl_gnu_thread \
    -lmkl_core \
    -fopenmp \
    -lpthread -lm -ldl \
    -o bin/cpu_intel_01_pacca_pb3
```

## Uso de ILP
```
python ILP.py --input dataset.csv
```

## Link de Collab
[Collab](https://colab.research.google.com/drive/14bjIRm0TNDywzpddDNzft9y4cKC842_g?usp=sharing)


