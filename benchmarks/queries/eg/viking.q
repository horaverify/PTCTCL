E[] y1 <= 40
E[] time <= 40
E[] y1 <= 40 && !Viking1.safe
E[] (Viking1.safe || Viking1.unsafe) && y1 <= 40
E[] !(Viking1.safe && Viking2.safe) && y1 <= 40
E[] y1 <= 40 && (y2 <= 30 || y2 >= 35) && !Viking1.safe
