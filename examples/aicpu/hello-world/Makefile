ASCEND_HOME_DIR := /usr/local/Ascend/cann
NPU_ARCH    := dav-2201

AICPU_FLAGS := -O2 \
               --cce-aicpu-L$(ASCEND_HOME_DIR)/lib64/device/lib64 \
               --cce-aicpu-laicpu_api \
               -I$(ASCEND_HOME_DIR)/include/ascendc/aicpu_api \
               -fPIC -c -g

.PHONY: all run clean


all: hello_world.aicpu main.asc
	bisheng $(AICPU_FLAGS) hello_world.aicpu -o hello_world.o
	bisheng --npu-arch=$(NPU_ARCH) main.asc -c -o main.o
	bisheng hello_world.o main.o -o demo

run:
	./demo

clean:
	rm -f *.o demo
