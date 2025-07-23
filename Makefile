# cleanlogs: get rid of contents of logs/
.PHONY: cleanlogs
cleanlogs:
	rm -rf logs/*
	@echo "Logs directory cleaned"
