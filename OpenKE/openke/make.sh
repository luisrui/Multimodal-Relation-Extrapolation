mkdir -p release
g++ ./base/Base.cpp -g -fPIC -shared -o ./release/Base.so -pthread -O2