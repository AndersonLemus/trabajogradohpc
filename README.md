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

g++ -O3 -march=znver4 -fopenmp -std=c++17 benchmark_cpu_amd.cpp \
        -I<OpenBLAS> -I<CSparse/Include> -I<fftw3/include> \
        -L<OpenBLAS> -L<CSparse/build> -L<fftw3/lib> \
        -lopenblas -lcsparse -lfftw3_threads -lfftw3 -lm -lpthread \
        -o benchmark_epyc

module load cuda
nvcc -O3 -arch=sm_52 -allow-unsupported-compiler benchmark_gpu_nvidia.cu -lcublas -lcusparse -lcufft -lnvml -lpthread -o benchmark_titanx
```

## Comando para manipulación y uso de ILP
```
python ILP.py --input dataset.csv --exact --solver CPLEX
```


