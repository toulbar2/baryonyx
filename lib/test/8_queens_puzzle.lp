minimize
 + x_0_0 + x_0_1 + x_0_2 + x_0_3 + x_0_4 + x_0_5 + x_0_6 + x_0_7
 + x_1_0 + x_1_1 + x_1_2 + x_1_3 + x_1_4 + x_1_5 + x_1_6 + x_1_7
 + x_2_0 + x_2_1 + x_2_2 + x_2_3 + x_2_4 + x_2_5 + x_2_6 + x_2_7
 + x_3_0 + x_3_1 + x_3_2 + x_3_3 + x_3_4 + x_3_5 + x_3_6 + x_3_7
 + x_4_0 + x_4_1 + x_4_2 + x_4_3 + x_4_4 + x_4_5 + x_4_6 + x_4_7
 + x_5_0 + x_5_1 + x_5_2 + x_5_3 + x_5_4 + x_5_5 + x_5_6 + x_5_7
 + x_6_0 + x_6_1 + x_6_2 + x_6_3 + x_6_4 + x_6_5 + x_6_6 + x_6_7
 + x_7_0 + x_7_1 + x_7_2 + x_7_3 + x_7_4 + x_7_5 + x_7_6 + x_7_7

subject to
+ x_0_0 + x_0_1 + x_0_2 + x_0_3 + x_0_4 + x_0_5 + x_0_6 + x_0_7  = 1
+ x_1_0 + x_1_1 + x_1_2 + x_1_3 + x_1_4 + x_1_5 + x_1_6 + x_1_7  = 1
+ x_2_0 + x_2_1 + x_2_2 + x_2_3 + x_2_4 + x_2_5 + x_2_6 + x_2_7  = 1
+ x_3_0 + x_3_1 + x_3_2 + x_3_3 + x_3_4 + x_3_5 + x_3_6 + x_3_7  = 1
+ x_4_0 + x_4_1 + x_4_2 + x_4_3 + x_4_4 + x_4_5 + x_4_6 + x_4_7  = 1
+ x_5_0 + x_5_1 + x_5_2 + x_5_3 + x_5_4 + x_5_5 + x_5_6 + x_5_7  = 1
+ x_6_0 + x_6_1 + x_6_2 + x_6_3 + x_6_4 + x_6_5 + x_6_6 + x_6_7  = 1
+ x_7_0 + x_7_1 + x_7_2 + x_7_3 + x_7_4 + x_7_5 + x_7_6 + x_7_7  = 1
+ x_0_0 + x_1_0 + x_2_0 + x_3_0 + x_4_0 + x_5_0 + x_6_0 + x_7_0  = 1
+ x_0_1 + x_1_1 + x_2_1 + x_3_1 + x_4_1 + x_5_1 + x_6_1 + x_7_1  = 1
+ x_0_2 + x_1_2 + x_2_2 + x_3_2 + x_4_2 + x_5_2 + x_6_2 + x_7_2  = 1
+ x_0_3 + x_1_3 + x_2_3 + x_3_3 + x_4_3 + x_5_3 + x_6_3 + x_7_3  = 1
+ x_0_4 + x_1_4 + x_2_4 + x_3_4 + x_4_4 + x_5_4 + x_6_4 + x_7_4  = 1
+ x_0_5 + x_1_5 + x_2_5 + x_3_5 + x_4_5 + x_5_5 + x_6_5 + x_7_5  = 1
+ x_0_6 + x_1_6 + x_2_6 + x_3_6 + x_4_6 + x_5_6 + x_6_6 + x_7_6  = 1
+ x_0_7 + x_1_7 + x_2_7 + x_3_7 + x_4_7 + x_5_7 + x_6_7 + x_7_7  = 1

+ x_1_0 + x_2_1 + x_3_2 + x_4_3 + x_5_4 + x_6_5 + x_7_6  <= 1
+ x_2_0 + x_3_1 + x_4_2 + x_5_3 + x_6_4 + x_7_5  <= 1
+ x_3_0 + x_4_1 + x_5_2 + x_6_3 + x_7_4  <= 1
+ x_4_0 + x_5_1 + x_6_2 + x_7_3  <= 1
+ x_5_0 + x_6_1 + x_7_2  <= 1
+ x_6_0 + x_7_1  <= 1
+ x_0_0 + x_1_1 + x_2_2 + x_3_3 + x_4_4 + x_5_5 + x_6_6 + x_7_7  <= 1
+ x_0_1 + x_1_2 + x_2_3 + x_3_4 + x_4_5 + x_5_6 + x_6_7  <= 1
+ x_0_2 + x_1_3 + x_2_4 + x_3_5 + x_4_6 + x_5_7  <= 1
+ x_0_3 + x_1_4 + x_2_5 + x_3_6 + x_4_7  <= 1
+ x_0_4 + x_1_5 + x_2_6 + x_3_7  <= 1
+ x_0_5 + x_1_6 + x_2_7  <= 1
+ x_0_6 + x_1_7  <= 1

+ x_1_0 + x_0_1  <= 1
+ x_2_0 + x_1_1 + x_0_2  <= 1
+ x_3_0 + x_2_1 + x_1_2 + x_0_3  <= 1
+ x_4_0 + x_3_1 + x_2_2 + x_1_3 + x_0_4  <= 1
+ x_5_0 + x_4_1 + x_3_2 + x_2_3 + x_1_4 + x_0_5  <= 1
+ x_6_0 + x_5_1 + x_4_2 + x_3_3 + x_2_4 + x_1_5 + x_0_6  <= 1
+ x_7_0 + x_6_1 + x_5_2 + x_4_3 + x_3_4 + x_2_5 + x_1_6 + x_0_7  <= 1
+ x_7_1 + x_6_2 + x_5_3 + x_4_4 + x_3_5 + x_2_6 + x_1_7  <= 1
+ x_7_2 + x_6_3 + x_5_4 + x_4_5 + x_3_6 + x_2_7  <= 1
+ x_7_3 + x_6_4 + x_5_5 + x_4_6 + x_3_7  <= 1
+ x_7_4 + x_6_5 + x_5_6 + x_4_7  <= 1
+ x_7_5 + x_6_6 + x_5_7  <= 1
+ x_7_6 + x_6_7  <= 1

binary
  x_0_0  x_0_1  x_0_2  x_0_3  x_0_4  x_0_5  x_0_6  x_0_7
  x_1_0  x_1_1  x_1_2  x_1_3  x_1_4  x_1_5  x_1_6  x_1_7
  x_2_0  x_2_1  x_2_2  x_2_3  x_2_4  x_2_5  x_2_6  x_2_7
  x_3_0  x_3_1  x_3_2  x_3_3  x_3_4  x_3_5  x_3_6  x_3_7
  x_4_0  x_4_1  x_4_2  x_4_3  x_4_4  x_4_5  x_4_6  x_4_7
  x_5_0  x_5_1  x_5_2  x_5_3  x_5_4  x_5_5  x_5_6  x_5_7
  x_6_0  x_6_1  x_6_2  x_6_3  x_6_4  x_6_5  x_6_6  x_6_7
  x_7_0  x_7_1  x_7_2  x_7_3  x_7_4  x_7_5  x_7_6  x_7_7

end
