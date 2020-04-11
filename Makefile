SRC_DIR = ./fastsk/src
BIN_DIR = ./bin

all:
	+$(MAKE) -C $(SRC_DIR)
	mkdir -p ./bin
	cp $(SRC_DIR)/fastsk $(BIN_DIR)/fastsk
	rm $(SRC_DIR)/fastsk
	@echo fastsk executable installed in the ./bin directory

clean:
	cd $(SRC_DIR); $(RM) *.o *~ fastsk
	cd $(BIN_DIR); $(RM) fastsk
	