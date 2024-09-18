# Build example using the .c files in this directory and the static libartl.a
lowlevelapi_example: lowlevelapi_example.c lowlevelapi.c libartl/libartl.a
	gcc $^ -o $@

# Build LLAPI shared library file for Python wrapper
lowlevelapi: lowlevelapi.c libartl/libartl.a
	gcc -fPIC -shared -o lowlevelapi.so lowlevelapi.c libartl/libartl.a

interface: interface.c lowlevelapi.c udp.c libartl/libartl.a
	gcc $^ -o $@
	gcc -fPIC -shared -o interface.so interface.c lowlevelapi.c udp.c libartl/libartl.a -pthread

# Build the static ARTL library
libartl/libartl.a:
	$(MAKE) -C libartl libartl.a

# Clean outputs from both this directory and libartl
clean:
	rm -f lowlevelapi_example
	rm -f lowlevelapi.so
	rm -f interface.so
	$(MAKE) -C libartl clean

.PHONY: clean
