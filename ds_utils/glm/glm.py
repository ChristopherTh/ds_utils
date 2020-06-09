from pathlib import Path
from datetime import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy import stats
import webbrowser



df = df_autos

glm_func('city_mpg ~ weight', data = df , glm_type = 'gaussian')

def glm_func(formular, data, glm_type):

    # scripting
    file_path = Path.cwd()

    exp = file_path.joinpath("experiments")

    if not exp.exists():
        exp.mkdir()
        
    new_folder = exp.joinpath(datetime.today().strftime("%d.%m.%Y um %H:%M:%S Uhr"))

    new_folder.mkdir()

    # selecting model
    if glm_type == 'gamma':
        fam = sm.families.Gamma(sm.families.links.log)
    elif glm_type == 'poisson':
        fam = sm.families.Poisson(sm.families.links.log)
    elif glm_type == 'gaussian':
        fam = sm.families.Gaussian()


    model = smf.glm(formula=formular, data=data, family=fam).fit()

    with open(new_folder.joinpath("Output.txt"), "w") as output:
        output.write(str(model.summary()))


    fig, ax = plt.subplots()
    resid = model.resid_deviance.copy()
    resid_std = stats.zscore(resid)
    ax.hist(resid_std, bins=25)
    ax.set_title('Histogram of standardized deviance residuals')
    fig.savefig(new_folder.joinpath("plot.png"))

    #webbrowser.open(str(new_folder.joinpath("Output.txt")))

