FROM julia:1.4.1

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

USER root

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-dev \
    python3-distutils \
    curl \
    ca-certificates \
    git \
    libgconf-2-4 \
    xvfb \
    libgtk-3-0 \
    dvipng \
    texlive-latex-recommended  \
    zip \
    htop \
    libxt6 libxrender1 libxext6 libgl1-mesa-glx libqt5widgets5 # GR


# Switch default user
USER ${NB_USER}
ENV PATH=${HOME}/.local/bin:$PATH

RUN curl -kL https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip3 install \
    jupyter \
    jupytext \
    ipywidgets \
    jupyter-contrib-nbextensions \
    jupyter-nbextensions-configurator


RUN jupyter notebook --generate-config && \
    echo "\
c.ContentsManager.default_jupytext_formats = 'ipynb,jl'\n\
c.NotebookApp.contents_manager_class = 'jupytext.TextFileContentsManager'\n\
c.NotebookApp.open_browser = False\n\
" >> ${HOME}/.jupyter/jupyter_notebook_config.py

# prepare to install extension
RUN jupyter contrib nbextension install --user && \
    jupyter nbextensions_configurator enable --user && \
    # enable extensions what you want
    jupyter nbextension enable select_keymap/main && \
    jupyter nbextension enable highlight_selected_word/main && \
    jupyter nbextension enable toggle_all_line_numbers/main && \
    jupyter nbextension enable varInspector/main && \
    jupyter nbextension enable toc2/main && \
    jupyter nbextension enable equation-numbering/main && \
    jupyter nbextension enable livemdpreview/livemdpreview && \
    jupyter nbextension enable execute_time/ExecuteTime && \
    echo Done


RUN mkdir -p ${HOME}/.julia/config && \
    echo '\
# set environment variables\n\
ENV["PYTHON"]=Sys.which("python3")\n\
ENV["JUPYTER"]=Sys.which("jupyter")\n\
\n\
import Pkg\n\
let\n\
    pkgs = ["Revise","OhMyREPL"]\n\
    for pkg in pkgs\n\
        if Base.find_package(pkg) === nothing\n\
            Pkg.add(pkg)\n\
        end\n\
    end\n\
end\n\
using OhMyREPL \n\
enable_autocomplete_brackets(false) \n\
atreplinit() do repl\n\
    try\n\
        @eval using Revise\n\
        @async Revise.wait_steal_repl_backend()\n\
    catch e\n\
        @warn(e.msg)\n\
    end\n\
end\n\
\n\
' >> ${HOME}/.julia/config/startup.jl && cat ${HOME}/.julia/config/startup.jl

# Install Julia Package
RUN julia -E 'using Pkg; \
Pkg.add(["Atom", "Juno"]); \
Pkg.add(["Pkg"]); \
Pkg.add(["BinaryProvider"]); \
Pkg.add(["Random", "Optim", "BlackBoxOptim", "JuMP", "Ipopt", "ForwardDiff"]);\
Pkg.add(["OhMyREPL", "Revise"]); \
Pkg.add(["Plots", "GR", "PyCall", "DataFrames"]); \
Pkg.add("PackageCompiler"); \
Pkg.add(["Documenter", "Literate", "Weave", "Franklin", "NodeJS"]); \
Pkg.add(["Plotly", "PlotlyJS", "ORCA"]); \
Pkg.add(["IJulia", "Interact", "WebIO"]); \
Pkg.precompile() \
'

# suppress warning for related to GR backend
ENV GKSwstype=100

# Do Ahead of Time Compilation using PackageCompiler
# For some technical reason, we switch default user to root then we switch back again
USER root
RUN julia --trace-compile="traced.jl" -e '\
        using OhMyREPL, Revise, Plots, PyCall, DataFrames; \
    ' && \
    julia -e '\
        using PackageCompiler; \
        PackageCompiler.create_sysimage([:OhMyREPL, :Revise, :Plots, :GR, :PyCall, :DataFrames]; precompile_statements_file="traced.jl", replace_default=true) \
    ' && \
    rm traced.jl

COPY ./.statements /tmp

RUN mkdir -p /sysimages && julia -e '\
    using PackageCompiler; PackageCompiler.create_sysimage(\
        [:OhMyREPL, :Revise, :Plots, :GR, :PyCall, :DataFrames, :IJulia], \
        precompile_statements_file="/tmp/ijuliacompile.jl", \
        precompile_execution_file="/tmp/execution.jl", \
        sysimage_path="/sysimages/ijulia.so", \
    ) \
    '

# Make NB_USER Occupy julia binary
RUN chown -R ${NB_UID} /usr/local/julia /sysimages
# Swich user again to NB_USER
USER ${NB_USER}

# Pkgs with respect to Jupyter
# When initialize jupyter notebook ...
# Install kernel so that `JULIA_PROJECT` should be ${HOME}
RUN jupyter nbextension uninstall --user webio/main && \
    jupyter nbextension uninstall --user webio-jupyter-notebook && \
    julia -e '\
              using IJulia, WebIO ; \
              WebIO.install_jupyter_nbextension(); \
              envhome=ENV["HOME"]; \
              installkernel("Julia", "--project=$envhome", "-J/sysimages/ijulia.so");\
              ' && \
    echo "Done"

# Make sure the contents of our repo are in ${HOME}
WORKDIR ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

RUN pip install -r requirements.txt
ENV JULIA_PROJECT=${HOME}
# Initialize Julia package using /work/Project.toml
RUN julia --project=${HOME} -e 'using Pkg; \
Pkg.instantiate(); \
Pkg.precompile()' && \
# Check Julia version \
julia -e 'using InteractiveUtils; versioninfo()'

# For Jupyter Notebook
EXPOSE 8888
# For Http Server
EXPOSE 8000
