HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)

all: kernels.code launchKernelHcc.hip.out 

launchKernelHcc.hip.out: launchKernelHcc.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

defaultDriver.hip.out: defaultDriver.cpp
	$(HIPCC)  $(HIPCC_FLAGS) $< -o $@

kernels.code: kernels.cpp
	$(HIPCC)  --genco  $(GENCO_FLAGS) $^ -o $@

clean:
	rm -f *.code *.out
