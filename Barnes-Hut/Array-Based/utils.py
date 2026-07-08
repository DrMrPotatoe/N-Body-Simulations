def split_integer(total, n) -> list[int]:
    ''' Split an integer into n bins, all with integer values'''
    base, rem = divmod(total, n)
    return [base + (i < rem) for i in range(n)]