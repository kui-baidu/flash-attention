NVCC=nvcc

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
     --threads=4 

libattn: cus
	$(NVCC) -shared -Xcompiler="-fPIC" -o libattn.so *.o

cus: clean sync
	#$(NVCC) $(NVCCARG) -c fmha_api.cpp
	$(NVCC) $(NVCCARG) -c src/fmha_fwd_hdim32.cu
	$(NVCC) $(NVCCARG) -c src/fmha_fwd_hdim64.cu
	$(NVCC) $(NVCCARG) -c src/fmha_fwd_hdim128.cu
	$(NVCC) $(NVCCARG) -c src/fmha_bwd_hdim32.cu
	$(NVCC) $(NVCCARG) -c src/fmha_bwd_hdim64.cu
	$(NVCC) $(NVCCARG) -c src/fmha_bwd_hdim128.cu
	$(NVCC) $(NVCCARG) -c src/fmha_block_fprop_fp16_kernel.sm80.cu
	$(NVCC) $(NVCCARG) -c src/fmha_block_dgrad_fp16_kernel_loop.sm80.cu

sync:
	git submodule update --init cutlass

clean:
	rm -rf *.so
	rm -rf *.o
