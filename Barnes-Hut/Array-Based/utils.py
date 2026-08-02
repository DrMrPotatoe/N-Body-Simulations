def split_integer(total: int, n: int) -> list[int]:
    ''' Split an integer into n bins, all with integer values'''
    if n < 1: return [total]
    base, rem = divmod(total, n)
    return [base + (i < rem) for i in range(n)]