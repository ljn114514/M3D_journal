import os

file = open('list_test_seq.txt','w')
label = -2
seq_id = -1

num = 0
for line in open('list_mars_test.txt'):
	line = line.split()
	if label != int(line[1]):
		label = int(line[1])
		seq_id = int(line[0][12:16])
		seq = line[0][0:16]+ ' ' + str(label) + '\n'
		file.write(seq)
		num = num + 1
	elif seq_id != int(line[0][12:16]):
		label = int(line[1])
		seq_id = int(line[0][12:16])
		label = int(line[1])
		seq = line[0][0:16]+ ' ' + str(label) + '\n'
		file.write(seq)
		num = num + 1
print num