#!/usr/bin/env python3
import sys

if __name__ == "__main__":
    args = [ i for i in sys.argv]
    
print(f'cmd.dist("chain {args[1]} and resi {args[2]} and name {args[3]}", "chain {args[4]} and resi {args[5]} and name {args[6]}")')

