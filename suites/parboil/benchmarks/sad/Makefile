DATASET=short
PROG=lbm
SUITE_NAME=parboil
OPTION=none

all: clean $(PROG) install

$(PROG):
	cd ../../ && \
	  ./parboil compile $(PROG) cuda $(OPTION) && \
	  cd -
clean:
	cd ../../ && \
	./parboil clean $(PROG) cuda && \
	cd -
test:
	cd ../../ && \
	./parboil run $(PROG) cuda $(DATASET) && \
	cd -

golden:
	cd ../../ && \
	./parboil run $(PROG) cuda $(DATASET) && \
	cd -

install:
	mkdir -p $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)
	cp build/cuda_default/$(PROG) $(SASSIFI_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

clobber: clean
	rm -fr run/* sassifi-inst-counts.txt
