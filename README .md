# Loan Default Risk Predictor

This project predicts how likely someone is to default on a loan, meaning they stop paying it back before it's fully repaid. It walks through the whole journey of a real machine learning project: starting with messy real world data, cleaning it up, teaching a model to spot risky applicants, explaining why the model made each decision, and finally wrapping it all up in a small web service anyone can send requests to.

## What does "default" actually mean?

When someone takes out a loan, they agree to pay it back over time, usually in monthly installments. If they stop paying and the loan is never fully repaid, that's called a default. Lenders lose money when this happens, so being able to estimate how likely a new applicant is to default helps them make smarter, fairer lending decisions.

## What does this project actually do?

Given some basic facts about a loan applicant, like their income, credit score, and how much debt they're already carrying, the model estimates a probability of default. For example it might say there's a 14 percent chance this person defaults, which a lender could use as one input among others when deciding whether to approve a loan and at what terms.

To be clear, this project doesn't decide whether to approve someone. It produces a risk score that could feed into that decision, similar to how a credit score itself is just one number that informs a bigger decision, not the whole decision.

## Where the data came from

The dataset was sourced from Kaggle, and originally comes from Lending Club, a real peer to peer lending company. It contains about 9,600 real loan records. Because it's real data rather than something artificially generated, it came with genuine messiness that had to be cleaned up before it could be used, which is described below.

## Understanding the columns

Here is what each piece of information about an applicant actually represents, explained in plain terms.

Annual income is simply how much money the applicant earns in a year.

Dti stands for debt to income ratio. It's a percentage showing how much of someone's income is already going toward paying off debt. Someone with a dti of 30 is spending 30 percent of their income just on existing debt payments, which is a meaningful risk signal, since it leaves less room to handle a new loan.

Fico is a credit score, ranging roughly from 300 to 850, that summarizes how reliably someone has paid back debts in the past. Higher is better. Banks and lenders in the United States rely heavily on this number.

Revol.bal, short for revolving balance, is how much debt someone is currently carrying on things like credit cards, where the balance can go up and down month to month rather than being a fixed loan amount.

Revol.util, or revolving utilization, is a percentage showing how much of their available credit someone is actually using. If someone has a 10,000 dollar credit limit and has used 8,000 dollars of it, their utilization is 80 percent, which usually signals financial strain, since it means they're close to maxing out what's available to them.

Inq.last.6mths counts how many times a lender has checked this person's credit in the past six months. A lot of recent inquiries can mean someone is actively searching for credit, sometimes because they're in a tight financial spot.

Delinq.2yrs counts how many times in the past two years the person was seriously late on a payment, typically 30 days or more overdue. This is direct evidence of past repayment trouble.

Pub.rec counts serious financial red flags on public record, such as bankruptcies or tax liens. These are rare but significant when present.

Years with credit line tells us how long the applicant has had any form of credit history at all. Generally, a longer credit history gives a clearer, more trustworthy picture of someone's financial habits.

Purpose is simply why the applicant wants the loan, such as consolidating existing debt, paying for a major purchase, or covering medical or educational expenses. Some purposes tend to be riskier than others in practice.

## The messy parts of the data, and how they were handled

Real data is rarely clean, and part of this project was learning to notice and fix problems rather than ignore them.

Some rows were missing several fields at once, and it turned out this wasn't random. All of these rows belonged to a specific group of loans that hadn't met the lender's usual approval standards. Rather than guessing values for these rows, they were left as genuinely missing and the model was allowed to learn from the fact that they were missing, since that in itself carried meaning.

Some columns that should have been pure numbers had a few rows where someone had typed a word instead, such as writing the word six instead of the number 6, or yes and no instead of an actual count. Each of these had to be tracked down individually and corrected sensibly.

A few rows had impossible values, like a credit score of 1812, which doesn't exist on the real 300 to 850 scale. Looking closely, it became clear an extra digit had accidentally been added, and the true value could be recovered rather than thrown away.

## An important decision: what the model is and isn't allowed to know

Two pieces of information in the original data, the interest rate and whether the loan met the lender's standard policy, are actually decided by the lender only after they've already assessed how risky the applicant is. In other words, those two columns are a result of the very judgment this model is trying to make, not information a brand new applicant would have going in. Including them would let the model quietly cheat by peeking at the answer, so they were deliberately left out.

## Choosing a model

Three different modeling approaches were tried and compared honestly, rather than assuming a fancier method would automatically win.

Logistic regression, the simplest of the three, actually performed the best.

XGBoost, a more complex and popular method, came in close behind after careful tuning.

Random forest performed about the same as XGBoost.

Because the simplest model matched or beat the more complex ones, this told us the relationship between an applicant's details and their risk of default is fairly straightforward, and there wasn't much extra benefit to using a more complicated model here. Logistic regression was chosen as the final model, since it performed just as well while being simpler, faster, and easier to explain to a real person.

## Explaining individual predictions

A single risk score isn't very satisfying on its own. If someone is told they're high risk, they deserve to know why. This project uses a method called SHAP to break down every single prediction into the specific factors that pushed it up or down, and by how much. For example, a prediction might be explained by saying a low credit score pushed this person's risk up significantly, while having no recent credit inquiries pushed it back down a little. This mirrors the kind of specific, honest reasoning that real lenders are often required to give applicants.

## Using the API

The trained model is wrapped in a small web service built with FastAPI, so anyone, or any other piece of software, can send it an applicant's details and get back a prediction.

Sending a request like this, describing a realistic applicant:

```json
{
  "annual_income": 60000,
  "dti": 12.0,
  "fico": 750,
  "revol_bal": 8000,
  "revol_util": 25.0,
  "inq_last_6mths": 0,
  "delinq_2yrs": 0,
  "pub_rec": 0,
  "years_with_cr_line": 10.0,
  "purpose": "debt_consolidation"
}
```

Returns a response like this:

```json
{
  "default_probability": 0.14,
  "risk_tier": "Low Risk",
  "explanation": [
    {"feature": "fico", "value": 750, "contribution": -0.31},
    {"feature": "dti", "value": 12.0, "contribution": -0.08}
  ]
}
```

Notice that whoever calls this API only needs to know plain, real world facts about the applicant. All the behind the scenes math, like converting income into the format the model actually expects, happens automatically inside the API itself.

## Project layout

```
loan-default-predictor/
data/            raw and cleaned datasets (not included in the repo, generated by running the notebook)
notebooks/
    train_model.ipynb     cleans the data, compares models, saves the final trained model
    explain_model.ipynb   generates SHAP explanations for individual predictions
app/
    main.py                the FastAPI web service
models/          trained model files, created after running train_model.ipynb
requirements.txt
```

## Running this yourself

First, install the required packages:

```
pip install -r requirements.txt
```

Then open notebooks/train_model.ipynb and run every cell from top to bottom. This cleans the data, trains and compares the models, and saves the final chosen model into the models folder.

Finally, start the API:

```
uvicorn app.main:app --reload
```

Then open your browser to http://127.0.0.1:8000/docs to try it out directly, no separate frontend needed.

## What this project shows

Real world data problems, like values missing in a pattern rather than randomly, hidden text typed into number columns, and outliers with a recoverable explanation, need real investigation rather than a quick blanket fix.

Whether a piece of information is fair to use in a model depends on when that information would actually be available in real life, not simply whether including it happens to improve accuracy. This project deliberately left out the interest rate and the lender's policy flag for exactly this reason.

A more complicated model isn't automatically a better one. Comparing a simple logistic regression against XGBoost and a random forest, honestly and side by side, showed the simplest option actually performed just as well, if not slightly better, and it was chosen for that reason rather than for being the fanciest.

Finally, a model that can explain its own reasoning is far more trustworthy and useful than one that only produces a bare number, especially in a sensitive area like lending, where someone denied a loan deserves to know why.
