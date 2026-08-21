from quanllm_harness.interfaces.timing import format_elapsed


def test_elapsed_time_format_is_shared_and_zero_padded():
    assert format_elapsed(-1) == "00小时00分钟00秒"
    assert format_elapsed(5.9) == "00小时00分钟05秒"
    assert format_elapsed(3661.8) == "01小时01分钟01秒"
