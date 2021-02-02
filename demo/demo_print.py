import numpy as np

tk=np.arange(5)
lk=np.random.permutation(5)
dist=np.random.rand(5)

headers = ['True centroid', 'Closest learned centroid', 'Distance']
width = [len(h)+5 for h in headers]
title = ['{: >{width}}'.format(h, width=len(h) + 5) for h in headers]
bar = ['{: >{width}}'.format('-'*len(h), width=len(h)+5) for h in headers]
print(''.join(title))
print(''.join(bar))

for i in tk:
    print('{: >{wtk}}{: >{wlk}}{: >{wdist}{ftype}}'.format(i, lk[i], dist[i],
                                                    wtk=width[0], wlk=width[1],
                                                    wdist=width[2], ftype='G'))
