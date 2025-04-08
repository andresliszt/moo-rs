.DEFAULT_GOAL := help

#-------------------------------------------------
# Root Makefile for my-project
#-------------------------------------------------

.PHONY: help pymoors-% moors-%

# Proxy targets into subdirectories
pymoors-%:  ## Run target in pymoors/Makefile
	@$(MAKE) -C pymoors $*

moors-%:  ## Run target in moors/Makefile
	@$(MAKE) -C moors $*

# Help message showing usage and available targets from sub-Makefiles
help:
	@echo "Usage: make [pymoors-<target> | moors-<target>]"
	@echo
	@echo "pymoors targets (see pymoors/Makefile):"
	@sed -nE 's/^([a-zA-Z0-9_-]+):.*## (.*)/  pymoors-\1: \2/p' pymoors/Makefile
	@echo
	@echo "moors targets (see moors/Makefile):"
	@sed -nE 's/^([a-zA-Z0-9_-]+):.*## (.*)/  moors-\1: \2/p' moors/Makefile
