from kernel_tool import KernelBenchTool

def test_grid_too_large():
    tool = KernelBenchTool()
    big = "void k() { k<<<(2048,1,1)>>>( ); }"  # exceeds 1024 limit
    small = "void k() { k<<<(32,1,1)>>>( ); }"
    assert tool._grid_too_large(big) is True
    assert tool._grid_too_large(small) is False
