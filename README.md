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
srun -p amd -w exadell --nodes=1 --ntasks=1 --cpus-per-task=8 --time=02:00:00 --pty bash -i


srun -p gpu_titan -w felix --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=02:00:00 --pty bash -i
```

## Ejecutar el .sh según corresponda
```
./source setup_env.sh (Intel)
source ./setup_env_cuda.sh (Nvidia)
```

## Compilación
```
g++ -O3 -march=native -std=c++17 src/hpc_dataset.cpp -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -o bin/hpc_dataset


//AMD - CPU
g++ -O3 -march=znver4 -fopenmp -std=c++17 src/benchmark_cpu_amd.cpp \
        -I /home/jsgalvisb/proyecto/libs/OpenBLAS -I /home/jsgalvisb/proyecto/libs/SuiteSparse-7.7.0/CSparse/Include -I /home/jsgalvisb/proyecto/libs/fftw-3.3.10/api \
        -L /home/jsgalvisb/proyecto/libs/OpenBLASL -L /home/jsgalvisb/proyecto/libs/SuiteSparse-7.7.0/CSparse/build -L /home/jsgalvisb/proyecto/libs/fftw-3.3.10/.libs \
        -lopenblas -lcsparse -lfftw3_threads -lfftw3 -lm -lpthread \
        -o bin/benchmark_epyc

nvcc -O3 -arch=sm_52 -allow-unsupported-compiler benchmark_gpu_nvidia.cu -L/usr/lib64 -lnvidia-ml -lcublas -lcusparse -lcufft -lpthread -o benchmark_titanx
```

## Comando para manipulación y uso de ILP
```
python ILP.py --input dataset.csv --exact --solver CPLEX
```


