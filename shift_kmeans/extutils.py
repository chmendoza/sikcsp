'''
Utilities external to the functionality of the main algorithm
'''


def print_table_header(headers):
    '''
    Print headers and horizontal bar
    '''

    title = ['{: >{width}}'.format(h, width=len(h) + 5) for h in headers]
    hbar = ['{: >{width}}'.format('-' * len(h), width=len(h) + 5)
            for h in headers]

    print(''.join(title))
    print(''.join(hbar))
