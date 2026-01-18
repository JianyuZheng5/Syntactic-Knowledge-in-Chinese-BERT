with open("UD_all.txt", "r", encoding='utf-8') as fp:
    lines = fp.readlines()

    
Total = 0
correct = 0      
for line in lines:
    line = line.strip().split()
    if len(line) != 10:
        continue
    
    idd = line[0]
    if idd == "#":
        continue
    head = line[6]
    rel = line[7]
    
    if rel in ['WP', 'punct']:
        continue
    if int(idd)+1 == int(head):
        correct += 1
    Total += 1
    
print(correct/Total)
