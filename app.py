from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

app = Flask(__name__)
app.secret_key = "some_secret"

def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate a random dataset X of size N with values between 0 and 1
    X = np.random.uniform(0, 1, N)
    
    # Generate Y with the specified beta0, beta1, mu, and error term
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)
    
    # Fit a linear regression model to X and Y
    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Generate a scatter plot with regression line
    plt.figure()
    plt.scatter(X, Y, color='blue', alpha=0.5)
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red')
    plt.xlabel("X")
    plt.ylabel("Y")
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()
    
    # Run simulations to generate slopes and intercepts
    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.uniform(0, 1, N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)
        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)
    
    # Plot histograms of slopes and intercepts
    plt.figure()
    plt.hist(slopes, bins=20, alpha=0.7, label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.7, label="Intercepts")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()
    
    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = np.mean(np.abs(slopes) >= abs(slope))
    intercept_extreme = np.mean(np.abs(intercepts) >= abs(intercept))
    
    return X, Y, slope, intercept, plot1_path, plot2_path, slope_more_extreme, intercept_extreme, slopes, intercepts


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # Retrieve parameters from form submission
    N = int(request.form["N"])
    mu = float(request.form["mu"])
    sigma2 = float(request.form["sigma2"])
    beta0 = float(request.form["beta0"])
    beta1 = float(request.form["beta1"])
    S = int(request.form["S"])

    # Generate data and initial plots
    (
        X, Y, slope, intercept, plot1, plot2,
        slope_extreme, intercept_extreme, slopes, intercepts
    ) = generate_data(N, mu, beta0, beta1, sigma2, S)

    # Store data in session
    session.update({
        "X": X.tolist(), "Y": Y.tolist(), "slope": slope,
        "intercept": intercept, "slopes": slopes, "intercepts": intercepts,
        "slope_extreme": slope_extreme, "intercept_extreme": intercept_extreme,
        "N": N, "mu": mu, "sigma2": sigma2, "beta0": beta0, "beta1": beta1, "S": S
    })

    # Render template with generated data and plots
    return render_template(
        "index.html", plot1=plot1, plot2=plot2, slope_extreme=slope_extreme,
        intercept_extreme=intercept_extreme, N=N, mu=mu, sigma2=sigma2,
        beta0=beta0, beta1=beta1, S=S,
    )


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    simulated_stats = np.array(slopes if parameter == "slope" else intercepts)
    observed_stat = slope if parameter == "slope" else intercept
    hypothesized_value = beta1 if parameter == "slope" else beta0

    if test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)
    else:  # "!="
        p_value = np.mean(np.abs(simulated_stats) >= abs(observed_stat))

    fun_message = "Rare event detected!" if p_value <= 0.0001 else None

    # Plot histogram for hypothesis testing
    plt.figure()
    plt.hist(simulated_stats, bins=20, alpha=0.7)
    plt.axvline(x=observed_stat, color="red", linestyle="--", label="Observed")
    plt.axvline(x=hypothesized_value, color="blue", linestyle="--", label="Hypothesized")
    plt.legend()
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.close()

    return render_template(
        "index.html", plot1="static/plot1.png", plot2="static/plot2.png",
        plot3=plot3_path, parameter=parameter, observed_stat=observed_stat,
        hypothesized_value=hypothesized_value, N=N, beta0=beta0, beta1=beta1,
        S=S, p_value=p_value, fun_message=fun_message,
    )


@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level")) / 100

    estimates = np.array(slopes if parameter == "slope" else intercepts)
    observed_stat = slope if parameter == "slope" else intercept
    true_param = beta1 if parameter == "slope" else beta0

    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates)

    margin = stats.t.ppf(1 - (1 - confidence_level) / 2, df=S-1) * std_estimate
    ci_lower, ci_upper = mean_estimate - margin, mean_estimate + margin
    includes_true = ci_lower <= true_param <= ci_upper

    # Plot confidence interval
    plt.figure()
    plt.scatter(range(S), estimates, color='gray', alpha=0.5)
    plt.axhline(y=mean_estimate, color='blue', label="Mean Estimate")
    plt.axhline(y=ci_lower, color='green', linestyle="--", label="CI Lower Bound")
    plt.axhline(y=ci_upper, color='green', linestyle="--", label="CI Upper Bound")
    plt.axhline(y=true_param, color='red', label="True Parameter")
    plt.legend()
    plot4_path = "static/plot4.png"
    plt.savefig(plot4_path)
    plt.close()

    return render_template(
        "index.html", plot1="static/plot1.png", plot2="static/plot2.png",
        plot4=plot4_path, parameter=parameter, confidence_level=int(confidence_level * 100),
        mean_estimate=mean_estimate, ci_lower=ci_lower, ci_upper=ci_upper,
        includes_true=includes_true, observed_stat=observed_stat, N=N,
        mu=mu, sigma2=sigma2, beta0=beta0, beta1=beta1, S=S,
    )


if __name__ == "__main__":
    app.run(debug=True)

