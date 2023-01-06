NVCC=nvcc
LIB=flashattn

NVCCARG=\
     -O3 \
     -Xcompiler="-fPIC" \
     -Xcompiler="-O3" \
     -Xcompiler="-DVERSION_GE_1_1" \
     -Xcompiler="-DVERSION_GE_1_3" \
     -Xcompiler="-DDVERSION_GE_1_5" \
     -gencode arch=compute_80,code=sm_80 \
     -U__CUDA_NO_HALF_OPERATORS__ \
     -U__CUDA_NO_HALF_CONVERSIONS__ \
     -I${PWD}/src \
     -I${PWD}/cutlass/include \
     --expt-relaxed-constexpr \
     --expt-extended-lambda \
     --use_fast_math \
     -DVERSION_GE_1_1 \
     -DVERSION_GE_1_3 \
     -DVERSION_GE_1_5 \
     --threads=10

lib: obj
	$(NVCC) -shared -Xcompiler="-fPIC" -o lib${LIB}.so *.o

obj: sync src/* flash_attn.cpp
	$(NVCC) $(NVCCARG) -c src/*.cu src/*.cpp flash_attn.cpp

sync: cutlass
	git submodule update --init cutlass

test:
	$(NVCC) -L${PWD} -ldl -l${LIB} flash_attn_test.cpp -o flash_attn_test
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD} ./flash_attn_test

clean:
	rm -rf *.so
	rm -rf *.o
	rm -rf flash_attn_test

rebuild:
	$(NVCC) $(NVCCARG) -c flash_attn.cpp
	$(NVCC) -shared -Xcompiler="-fPIC" -o lib${LIB}.so *.o
