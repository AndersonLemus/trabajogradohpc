# Trabajo de grado.

## Instalación MKL 

Agregar repositorio de Intel
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
Instalar la libreria MKL.
```
sudo dnf install intel-oneapi-mkl-devel
```


