
ASCEND_HOME_DIR := /usr/local/Ascend/cann
NPU_ARCH    := dav-2201

AICPU_FLAGS := -O2 \
               --cce-aicpu-L$(ASCEND_HOME_DIR)/lib64/device/lib64 \
               --cce-aicpu-laicpu_api \
               -I$(ASCEND_HOME_DIR)/include/ascendc/aicpu_api \
               -fPIC -c -g

hello_world.aicpu.o: hello_world.aicpu
	bisheng $(AICPU_FLAGS) $< -o $@

main.asc.o: main.asc
	bisheng --npu-arch=$(NPU_ARCH) main.asc -c -o main.asc.o

build: main.asc.o hello_world.aicpu.o
	bisheng hello_world.aicpu.o main.asc.o -o demo

run: build
	./demo

clean:
	rm -f *.o demo
