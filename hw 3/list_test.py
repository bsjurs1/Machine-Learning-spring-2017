a = [[1,4,"many"],[1,4,"many"],[2,3],[3,4]]

h = 0
tmp = None
for elem in a:
	if a.count(elem) > h:
		h = a.count(elem)
		tmp = elem

print h,tmp