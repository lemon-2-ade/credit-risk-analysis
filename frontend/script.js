document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('riskForm');
    
    // States
    const stateInitial = document.getElementById('initialState');
    const stateLoading = document.getElementById('loadingState');
    const stateResult = document.getElementById('resultState');
    const stateError = document.getElementById('errorState');
    
    // Result elements
    const riskLabel = document.getElementById('riskLabel');
    const probabilityVal = document.getElementById('probabilityVal');
    const riskFill = document.getElementById('riskFill');
    const riskDescription = document.getElementById('riskDescription');
    const resetBtn = document.getElementById('resetBtn');
    const retryBtn = document.getElementById('retryBtn');
    
    // Button UI
    const submitBtnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');

    function switchState(stateToShow) {
        [stateInitial, stateLoading, stateResult, stateError].forEach(el => {
            if (el === stateToShow) {
                el.classList.remove('hidden');
                // Little trick to trigger animations
                if (el === stateResult) {
                    riskFill.style.width = '0%';
                }
            } else {
                el.classList.add('hidden');
            }
        });
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Remove error states from form
        document.querySelectorAll('.error-border').forEach(el => el.classList.remove('error-border'));
        
        // Collect data
        const formData = {
            person_age: parseInt(document.getElementById('person_age').value),
            person_income: parseFloat(document.getElementById('person_income').value),
            person_home_ownership: document.getElementById('person_home_ownership').value,
            person_emp_length: parseFloat(document.getElementById('person_emp_length').value),
            loan_intent: document.getElementById('loan_intent').value,
            loan_grade: document.getElementById('loan_grade').value,
            loan_amnt: parseFloat(document.getElementById('loan_amnt').value),
            loan_int_rate: parseFloat(document.getElementById('loan_int_rate').value),
            cb_person_default_on_file: document.getElementById('cb_person_default_on_file').value,
            cb_person_cred_hist_length: parseInt(document.getElementById('cb_person_cred_hist_length').value)
        };

        // UI transition to Loading
        submitBtnText.classList.add('hidden');
        loader.classList.remove('hidden');
        document.getElementById('submitBtn').disabled = true;
        switchState(stateLoading);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            // Process results
            displayResult(data);

        } catch (error) {
            console.error('Error:', error);
            switchState(stateError);
        } finally {
            // Restore button
            submitBtnText.classList.remove('hidden');
            loader.classList.add('hidden');
            document.getElementById('submitBtn').disabled = false;
        }
    });

    function displayResult(data) {
        const probPct = Math.round(data.probability * 100);
        
        riskLabel.textContent = data.risk_category;
        probabilityVal.textContent = probPct;
        
        // Clear old themes
        stateResult.classList.remove('success-theme', 'warning-theme', 'danger-theme');
        
        if (data.risk_category === 'Low Risk') {
            stateResult.classList.add('success-theme');
            riskDescription.textContent = "This applicant shows strong financial indicators. Based on our AI models, this profile represents a low risk of default.";
            // The bar uses CSS gradient, so width handles the color showing
            setTimeout(() => { riskFill.style.width = Math.max(10, probPct) + '%'; }, 100);
        } else if (data.risk_category === 'Medium Risk') {
            stateResult.classList.add('warning-theme');
            riskDescription.textContent = "This profile exhibits mixed indicators. While not critically risky, there are some factors that may warrant a closer manual review.";
            setTimeout(() => { riskFill.style.width = probPct + '%'; }, 100);
        } else {
            stateResult.classList.add('danger-theme');
            riskDescription.textContent = "Caution advised. The financial and historical indicators for this profile suggest a high probability of default.";
            setTimeout(() => { riskFill.style.width = probPct + '%'; }, 100);
        }

        switchState(stateResult);
        
        // Mobile smooth scroll to results
        if (window.innerWidth < 900) {
            stateResult.scrollIntoView({ behavior: 'smooth' });
        }
    }

    resetBtn.addEventListener('click', () => {
        form.reset();
        switchState(stateInitial);
        window.scrollTo({top: 0, behavior: 'smooth'});
    });
    
    retryBtn.addEventListener('click', () => {
        switchState(stateInitial);
    });
});
