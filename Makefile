# cleanlogs: get rid of contents of logs/
.PHONY: cleanlogs
cleanlogs:
	rm -rf logs/*
	@echo "Logs directory cleaned"

# cleantimm: get rid of contents of configs/timm/
cleantimm:
	rm -rf configs/timm/*
	@echo "Timm configs directory cleaned"
