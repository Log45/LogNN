%include <std_string.i>
%include <std_vector.i>
%include <std_shared_ptr.i>

%template(SizeTVector) std::vector<size_t>;
%template(DoubleVector) std::vector<double>;
%template(SizeTVectorVector) std::vector<std::vector<size_t>>;
%template(VariableVector) std::vector<Variable>;
%template(ModulePtrVector) std::vector<std::shared_ptr<Module>>;
