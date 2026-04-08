# Trabajo de grado.

Para el uso y manejo del directorio al momen de entrar a sc3:
```
ssh guane
cd proyecto //Es la carpeta raíz del proyecto.
```

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
Ejecutar el siguiente .sh antes de ejecutar cualquier código relacionado a cpu (Intel)
```
./source setup_env.sh
```

## Comando usado para la generación gemm (CPU).
```
g++ src/gemm_l_mkl.cpp     -I/home/jsgalvisb/proyecto/src/papi/src     -L/home/jsgalvisb/proyecto/src/papi/src     -Wl,-rpath,/home/jsgalvisb/proyecto/src/papi/src     -lpapi -lmkl_rt -o gemm_exe
```

Para ejecutar cualquier archivo relacionado con cuda, es menester ejecutar el siguiente comando sin que el setup_env.sh haya sido ejecutado previamente
```
source ./setup_env_cuda.sh
```

## Comando usado para la generación de los kernels en CUDA.
```
nvcc -O3 -arch=sm_52 -allow-unsupported-compiler src/benchmark_nvidia.cu      -I/home/jsgalvisb/proyecto/src/papi/src      -L/home/jsgalvisb/proyecto/src/papi/src      -Xlinker -rpath,/home/jsgalvisb/proyecto/src/papi/src      -lcublas -lcusparse -lcufft -lnvml -lpapi -lpthread      -o bin/benchmark_titanx
```

## SRUN
```
CPU: srun -p gpu_titan -w felix --nodes=1 --ntasks=1 --cpus-per-task=8 --time=02:00:00 --pty bash -i
GPU: srun -p gpu_titan -w felix --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=02:00:00 --pty bash -i
```

## Compilación para Intel CPU
```
g++ -O3 -march=native -std=c++17 src/hpc_dataset.cpp \
    -I${MKLROOT}/include \
    -L${MKLROOT}/lib/intel64 \
    -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
    -lpthread -lm -ldl -o bin/hpc_dataset
```

## Comando para manipulación y uso de ILP
```
pip install cplex
python ILP.py --input dataset.csv --exact --solver CPLEX
```


