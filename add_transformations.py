# Set up sympy parser transformations for implicit multiplication
transformations = (standard_transformations + 
                  (implicit_multiplication_application,
                   convert_xor)) 