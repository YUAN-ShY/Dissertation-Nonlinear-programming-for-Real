var x >=0, <= 3.1415;;
var y;
var s >=0;

minimize obj: y;

subject to con1: 
	y = s - sin(x);


