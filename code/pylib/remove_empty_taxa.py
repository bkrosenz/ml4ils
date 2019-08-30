from sys import argv

with open(argv[1],'r') as f, open(argv[1]+'.clean','w') as fout:
    ln = f.readline()
    while ln:
        if ln.startswith('>'):
            ln2 = f.readline()
            if ln2!='\n':
                fout.write(ln)
                fout.write(ln2)
        else:
            fout.write(ln)
        ln = f.readline()        
