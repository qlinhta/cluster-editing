import sys
import signal

#https://www.optil.io/optilion/help/signals#python3
class Killer:
  exit_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit)
    signal.signal(signal.SIGTERM, self.exit)

  def exit(self,signum, frame):
    self.exit_now = True

killer = Killer()

#read the graph from stdin
for line in sys.stdin:
    print(line.strip())

#demo of capturing SIGTERM
#to see how it works, run:
#timeout 1 python3 main.py < instance.gr
#exit_now is True after 1 second (in the benchmark, it will be after 10 minutes)
while True:
    if killer.exit_now:
        break

print("fin")