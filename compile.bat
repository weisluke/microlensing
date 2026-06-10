nvcc -std=c++20 -arch=compute_75 -code="sm_75,compute_75" --use_fast_math -I ./include -O3 -D is_float -o ./bin/ipm_gpu ./src/main_ipm.cu
nvcc -std=c++20 -arch=compute_75 -code="sm_75,compute_75" --use_fast_math -I ./include -O3 -D is_double -o ./bin/ipm_gpu_double ./src/main_ipm.cu
nvcc -std=c++20 -arch=compute_75 -code="sm_75,compute_75" --use_fast_math -I ./include -O3 -D is_float -shared -o ./bin/lib_ipm.dll ./src/lib_ipm.cu 
nvcc -std=c++20 -arch=compute_75 -code="sm_75,compute_75" --use_fast_math -I ./include -O3 -D is_double -shared -o ./bin/lib_ipm_double.dll ./src/lib_ipm.cu 

nvcc -std=c++20 -arch=compute_75 -code="sm_75,compute_75" --use_fast_math -I ./include -O3 -D is_double -o ./bin/ccf_gpu ./src/main_ccf.cu
nvcc -std=c++20 -arch=compute_75 -code="sm_75,compute_75" --use_fast_math -I ./include -O3 -D is_double -shared -o ./bin/lib_ccf.dll ./src/lib_ccf.cu 

nvcc -std=c++20 -arch=compute_75 -code="sm_75,compute_75" --use_fast_math -I ./include -O3 -D is_double -o ./bin/ncc_gpu ./src/main_ncc.cu
nvcc -std=c++20 -arch=compute_75 -code="sm_75,compute_75" --use_fast_math -I ./include -O3 -D is_double -shared -o ./bin/lib_ncc.dll ./src/lib_ncc.cu 

nvcc -std=c++20 -arch=compute_75 -code="sm_75,compute_75" --use_fast_math -I ./include -O3 -D is_double -o ./bin/mif_gpu ./src/main_mif.cu
nvcc -std=c++20 -arch=compute_75 -code="sm_75,compute_75" --use_fast_math -I ./include -O3 -D is_double -shared -o ./bin/lib_mif.dll ./src/lib_mif.cu 

if not exist ./microlensing/lib/ mkdir ./microlensing/lib/
REM note that the paths here needs backwards slashes for windows
copy .\bin\*.dll .\microlensing\lib\
