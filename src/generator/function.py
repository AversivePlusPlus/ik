import sympy

class Variable:
    def __init__(self, symbol, typename = "double", default_value = None):
        self.symbol = symbol
        self.typename = typename
        self.default_value = default_value

    def __str__(self):
        template_def = "{decl} = {val}"
        template_decl = "{typename} {name}"
        decl = template_decl.format(name = self.symbol, typename = self.typename)
        if self.default_value == None:
            return decl
        else:
            return template_def.format(decl = decl, val = self.default_value)

class Param(Variable):
    def __init__(self, symbol, typename = "double", default_value = None):
        Variable.__init__(self, symbol, typename, default_value)
        
class Instruction:
    def __init__(self, code):
        self.code = code

    def __str__(self):
        return self.code

class ReturnInstruction(Instruction):
    def __init__(self, obj):
        template = "return {obj}"
        Instruction.__init__(self, template.format(obj = obj))
    
class Function:
    def __init__(self, name, return_type = "void", params = [], instructions = []):
        self.name = name
        self.return_type = return_type
        self.params = params
        self.instructions = instructions

    def __str__(self):
        template = "{return_type} {func_name}({params})\n{{\n{instructions}\n}}"
        params = ", ".join(list(map(str, self.params)))
        instrs = ";\n".join(list(map(str, self.instructions)))
        return template.format(return_type = self.return_type, func_name = self.name, params = params, instructions = instrs + ";")

class MainFunc(Function):
    def __init__(self, instructions = []):
        params = [Param("", "int"), Param("", "char**")]
        Function.__init__(self, "main", "int", params, instructions + [ReturnInstruction("0")])

def get_matrix_case(name, row, col, (rows, cols)):
    return "{name}[{i}]".format(name=name, i=row*cols+col)
        
def fill_matrix_instructions(name, matrix):
    instrs = []
    rows = len(matrix)
    cols = len(matrix[0])
    coord_expr_list = {}
    for row in range(0, rows):
        for col in range(0, cols):
            coord_expr_list[(row, col)] = sympy.simplify(sympy.sympify(matrix[row][col]))
    def count_expr(expr, ret = {}):
        if ret.has_key(expr):
            ret[expr] += 1
        else:
            ret[expr] = 1
        for a in expr.args:
            count_expr(a, ret)
    expr_count_list = {}
    for e in coord_expr_list:
        count_expr(coord_expr_list[e], expr_count_list)
    tmp_list = []
    tmp_count = 0
    for k, e in expr_count_list.iteritems():
        if e > 1 and sympy.Symbol != type(k) and k.args != ():
            s = sympy.Symbol("tmp" + str(tmp_count))
            instrs.append("double {tmp} = {val}".format(tmp = s, val = k))
            tmp_list += [(k, s)]
            tmp_count += 1
    for r, c in coord_expr_list:
        e = coord_expr_list[(r,c)].subs(tmp_list)
        if type(name) == list:
            instrs.append("{name} = {expr}".format(name=name[r][c], expr=e))
        else:
            instrs.append("{m} = {expr}".format(m=get_matrix_case(name, r,c, sympy.Matrix(matrix).shape), expr=e))
    return instrs

    
class MatrixReturningFunction(Function):
    def __init__(self, name, matrix, symbols):
        params = list(map(lambda s: Param(s, "double"), symbols))
        rows = len(matrix)
        cols = len(matrix[0])
        params += [
            Param("matrix_out", "double*"),
        ]
        instrs = []
        instrs += fill_matrix_instructions("matrix_out", matrix)
        Function.__init__(self, name, "void", params, instrs)

def get_transposed_jacobian(mat, sym):
    flat = []
    for row in mat:
        for case in row:
            flat.append(case)
    t_jacobian = []
    for s in sym:
        t_jacobian.append(list(map(lambda e: e.diff(s), flat)))
    return t_jacobian
        
class ForwardKinematicsFunction(MatrixReturningFunction):
    def __init__(self, chain):
        ids = range(0, chain.get_num_params())
        sym = list(map(lambda i: sympy.Symbol("q"+str(i)), ids))
        mat = chain.forward_kinematics(sym)
        MatrixReturningFunction.__init__(self, "{}_forward_kinematics".format(chain.name), mat, sym)

class TransposedJacobianFunction(MatrixReturningFunction):
    def __init__(self, chain):
        ids = range(0, chain.get_num_params())
        sym = list(map(lambda i: sympy.Symbol("q"+str(i)), ids))
        mat = chain.forward_kinematics(sym)
        t_jacobian = get_transposed_jacobian(mat, sym)
        MatrixReturningFunction.__init__(self, "{}_transposed_jacobian".format(chain.name), t_jacobian, sym)

class If(Instruction):
    def __init__(self, cond, instructions):
        template = "if({cond}) {{\n{instrs};\n}}"
        Instruction.__init__(self, template.format(cond = cond, instrs = ";\n".join(list(map(str, instructions)))))

        
class InverseKinematicsStepFunction(Function):
    def __init__(self, chain):
        name = "{}_inverse_kinematics_step".format(chain.name)
        params = [
            Param("target", "double*"),
        ]
        ids = range(0, chain.get_num_params())
        pnames = list(map(lambda i: "q"+str(i)+"_io", ids))
        symbols = list(map(lambda i: sympy.Symbol(i), pnames))
        params += list(map(lambda i: Param(i, "double*"), symbols))
        params += [Param("coeff", "double", "0.1")]
        ref_symbols = list(map(lambda i: sympy.Symbol("*"+i), pnames))
        matrix = chain.forward_kinematics(ref_symbols);
        matrix_shape = sympy.Matrix(matrix).shape
        t_jacobian = get_transposed_jacobian(matrix, ref_symbols)
        target = []
        coords = []
        for i in range(0, matrix_shape[0]):
            for j in range(0, matrix_shape[1]):
                target.append([sympy.Symbol(get_matrix_case("dist", i, j, matrix_shape))])
                coords.append((i,j))
        sympy_delta_matrix = sympy.Matrix(t_jacobian) * sympy.Matrix(target)
        delta_matrix = list(map(lambda e: [e], list(sympy_delta_matrix)))
        instrs = [
            "double dist[{}]".format(matrix_shape[0]*matrix_shape[1]),
            "{name}_forward_kinematics({s}, dist)".format(name=chain.name, s=", ".join(list(map(str, ref_symbols)))),
            "double delta[{}] = {{0}};".format(len(pnames)),
        ]
        instrs += list(map(lambda (i,j): "{d} = {t}-{d}".format(d=get_matrix_case("dist", i,j, matrix_shape), t=get_matrix_case("target", i,j, matrix_shape)), coords))
        instrs += fill_matrix_instructions(list(map(lambda i: ["delta[{}]".format(i)], range(0, len(pnames)))), delta_matrix)
        instrs += [
            "double sum = sqrt({})".format(" + ".join(list(map(lambda i: "delta[{i}]*delta[{i}]".format(i=i), range(0, len(delta_matrix)))))),
            "double dist_norm = sqrt({})".format(" + ".join(list(map(lambda (i,j): "{d}*{d}".format(d=get_matrix_case("dist", i,j, matrix_shape)), coords)))),
            If("sum != 0", [
                    "coeff *= dist_norm",
            ] + list(map(lambda (s,i): "{s} += (delta[{i}]/sum)*coeff".format(s=s, i=i), zip(ref_symbols, range(0, len(delta_matrix)))))),
            "return dist_norm"
        ]
        Function.__init__(self, name, "double", params, instrs)

class InverseKinematicsFunction(Function):
    def __init__(self, chain):
        name = "{}_inverse_kinematics".format(chain.name)
        params = [Param("target", "double*")]
        ids = range(0, chain.get_num_params())
        pnames = list(map(lambda i: "q"+str(i)+"_io", ids))
        symbols = list(map(lambda i: sympy.Symbol(i), pnames))
        params += list(map(lambda i: Param(i, "double*"), symbols))
        params += [
            Param("coeff", "double", "0.1"),
            Param("stop_dist", "double", "1"),
            Param("max_iter", "int", "100"),
        ]
        Function.__init__(self, name, "double", params, [
            "double d = stop_dist",
            For("int i = 0 ; i < max_iter && stop_dist <= d ; i++", [
                    "d = {name}_inverse_kinematics_step(target, {s}, coeff)".format(name=chain.name, s=", ".join(list(map(str, symbols)))),
            ]),
            "return d"
        ])
        
class Cpp:
    def __init__(self, defs):
        self.defs = defs

    def __str__(self):
        return "\n\n".join(list(map(str, self.defs)))

class For(Instruction):
    def __init__(self, conds, instructions):
        template = "for({conds}) {{\n{instrs};\n}}"
        Instruction.__init__(self, template.format(conds = conds, instrs = ";\n".join(list(map(str, instructions)))))

def get_fk_shape(chain):
    target = chain.forward_kinematics([0]*chain.get_num_params())
    return sympy.Matrix(target).shape
        
class KinematicsCpp(Cpp):
    def __init__(self, chain, additional_defs = []):
        (rows, cols) = get_fk_shape(chain)
        Cpp.__init__(self, [
            "#include <math.h>",
            "#define {name}_FORWARD_KINEMATICS_ROWS {val}".format(name=chain.name.upper(), val=rows),
            "#define {name}_FORWARD_KINEMATICS_COLS {val}".format(name=chain.name.upper(), val=cols),
            ForwardKinematicsFunction(chain),
            #TransposedJacobianFunction(chain),
            InverseKinematicsStepFunction(chain),
            InverseKinematicsFunction(chain),
        ] + additional_defs)
