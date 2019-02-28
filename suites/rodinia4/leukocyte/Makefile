include ../env.mk
TARGET = CUDA/leukocyte

all: clean $(TARGET) install 

$(TARGET):
	cd CUDA; make

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
	cp $(TARGET) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

test:
	./$(TARGET)  ../data/leukocyte/testfile.avi 2

golden:
	./$(TARGET) ../data/leukocyte/testfile.avi 2 >golden_stdout.txt 2>golden_stderr.txt


clean:
	cd CUDA; make clean
	rm -f meschach_lib/meschach.a

clobber:
	rm -f sassifi-inst-counts.txt golden* result.txt
