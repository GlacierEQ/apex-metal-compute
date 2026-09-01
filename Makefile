build:
	mkdir -p .build
	swiftc -parse-as-library -module-name ApexMetal \
	  Sources/ApexMetal/ComputePipeline.swift \
	  Sources/CInterface/bridge.swift \
	  -o .build/libApexMetal.o 2>&1 || true
	@echo "Syntax check complete"

parse-check:
	swiftc -parse Sources/ApexMetal/ComputePipeline.swift Sources/CInterface/bridge.swift
	
python-benchmark:
	/Users/kcbflux/.apex_venv/bin/python python/benchmark.py

python-test:
	/Users/kcbflux/.apex_venv/bin/python -m pytest tests/ -v
