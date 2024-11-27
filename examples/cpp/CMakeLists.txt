add_executable(example_hnsw example_hnsw.cpp)
target_link_libraries(example_hnsw vsag)

add_executable(example_fresh_hnsw example_fresh_hnsw.cpp)
target_link_libraries(example_fresh_hnsw vsag)

add_executable(example_diskann example_diskann.cpp)
target_link_libraries(example_diskann vsag_static)

add_executable(simple_hnsw simple_hnsw.cpp)
target_link_libraries(simple_hnsw vsag_static)

add_executable(simple_hgraph_sq8 simple_hgraph_sq8.cpp)
target_link_libraries(simple_hgraph_sq8 vsag_static)

add_executable(example_conjugate_graph example_conjugate_graph.cpp)
target_link_libraries(example_conjugate_graph vsag)

add_executable(custom_memory_allocator custom_memory_allocator.cpp)
target_link_libraries(custom_memory_allocator vsag)

add_executable(custom_logger custom_logger.cpp)
target_link_libraries(custom_logger vsag)

add_executable(vsag_ext_example vsag_ext_example.cpp)
target_link_libraries(vsag_ext_example vsag)

add_executable(bench_hnsw bench_hnsw.cpp)
target_link_libraries(bench_hnsw vsag)

add_executable(bench_diskann bench_diskann.cpp)
target_link_libraries(bench_diskann vsag)

add_executable(bench_hgraph bench_hgraph.cpp)
target_link_libraries(bench_hgraph vsag)

add_executable(bench_glass bench_glass.cpp)
target_link_libraries(bench_glass vsag)