Build test executable with `-g` flag (for attaching to debuggers) : `cmake -B build -DCMAKE_BUILD_TYPE=Debug`


Build test executable without any debug support : `cmake -B build -DCMAKE_BUILD_TYPE=Release`


Compile and run tests : `cmake --build build && ./build/neat_tests`
