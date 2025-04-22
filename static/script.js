
const loadingDiv = document.getElementById('loading');
let tableName;
let isRecording = false;
let mediaRecorder;
let audioChunks = [];
let originalButtonHTML = ""; // Store the original button HTML
window.onload = function () {
    // Reset variables
    const loadingDiv = document.getElementById('loading');
    let tableName = undefined;
    let isRecording = false;
    let mediaRecorder = undefined;
    let audioChunks = [];
    let originalButtonHTML = "";

    console.log("Variables reset on page reload");
};
async function loadTableColumns(table_name) {
    console.log("Loading columns for table:", table_name); // Debug statement
    const selectedTable = table_name;

    if (!selectedTable) {
        alert("Please select a table.");
        return;
    }

    try {
        const response = await fetch(`/get-table-columns/?table_name=${selectedTable}`);
        const data = await response.json();

        if (response.ok && data.columns) {
            const xAxisDropdown = document.getElementById("x-axis-dropdown");
            const yAxisDropdown = document.getElementById("y-axis-dropdown");

            // Reset dropdown options
            xAxisDropdown.innerHTML = '<option value="" disabled selected>Select X-Axis</option>';
            yAxisDropdown.innerHTML = '<option value="" disabled selected>Select Y-Axis</option>';

            // Populate options
            data.columns.forEach((column) => {
                const xOption = document.createElement("option");
                const yOption = document.createElement("option");

                xOption.value = column;
                xOption.textContent = column;

                yOption.value = column;
                yOption.textContent = column;

                xAxisDropdown.appendChild(xOption);
                yAxisDropdown.appendChild(yOption);
            });
        } else {
            alert("Failed to load columns.");
        }
    } catch (error) {
        console.error("Error loading table columns:", error);
        alert("An error occurred while fetching columns.");
    }
}
// Add event listener for "Enter" key press in the input field
document.getElementById("chat_user_query").addEventListener("keyup", function (event) {
    // Number 13 is the "Enter" key on the keyboard
    if (event.key === "Enter") {
        // Cancel the default action, if needed
        event.preventDefault();
        // Trigger the button element with a click
        sendMessage();
    }
});

async function generateChart() {
    const xAxisDropdown = document.getElementById("x-axis-dropdown");
    const yAxisDropdown = document.getElementById("y-axis-dropdown");
    const chartTypeDropdown = document.getElementById("chart-type-dropdown");

    const xAxis = xAxisDropdown.value;
    const yAxis = yAxisDropdown.value;
    const chartType = chartTypeDropdown.value;
    selectedTable = tableName;
    if (!selectedTable || !xAxis || !yAxis || !chartType) {
        alert("Please select all required fields.");
        return;
    }

    try {
        const response = await fetch("/generate-chart/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                table_name: selectedTable,
                x_axis: xAxis,
                y_axis: yAxis,
                chart_type: chartType,
            }),
        });

        const data = await response.json();
        if (response.ok && data.chart) {
            const chartContainer = document.getElementById("chart-container");
            chartContainer.innerHTML = ""; // Clear previous chart
            const chartDiv = document.createElement("div");
            chartContainer.appendChild(chartDiv);

            // Render the chart using Plotly
            Plotly.newPlot(chartDiv, JSON.parse(data.chart).data, JSON.parse(data.chart).layout);
        } else {
            alert(data.error || "Failed to generate chart.");
        }
    } catch (error) {
        console.error("Error generating chart:", error);
        alert("An error occurred while generating the chart.");
    }
}
function changePage(tableName, pageNumber, recordsPerPage) {
    if (pageNumber < 1) return;

    // Corrected: Using template literals to construct the URL
    fetch(`/get_table_data?table_name=${tableName}&page_number=${pageNumber}&records_per_page=${recordsPerPage}`)
        .then(response => response.json())
        .then(data => {
            const tableDiv = document.getElementById(`${tableName}_table`);
            if (tableDiv) {
                tableDiv.innerHTML = data.table_html;
            }
            updatePaginationLinks(tableName, pageNumber, data.total_pages, recordsPerPage);
        })
        .catch(error => {
            console.error('Error fetching table data:', error);
        });
}
function openTab(evt, tabName) {
    let i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

// Optionally, you can set the default active tab using JavaScript:
document.addEventListener("DOMContentLoaded", function () {
    document.getElementsByClassName("tablinks")[0].click(); // Open the first tab by default
});

function toggleDevMode() {
    const devModeToggle = document.getElementById('devModeToggle');
    const xlsxbtn = document.getElementById('xlsx-btn'); // Excel button container
    let interpBtn = document.getElementById('interpBtn'); // check if the buttons already exist
    let langchainBtn = document.getElementById('langchainBtn');

    if (devModeToggle.checked) {
        // Create buttons if they don't exist
        if (!interpBtn) {
            interpBtn = document.createElement('button');
            interpBtn.id = 'interpBtn';
            interpBtn.textContent = 'Interpretation Prompt';
            interpBtn.className = 'dev-mode-btn';  // Add class for styling
            xlsxbtn.appendChild(interpBtn);
            interpBtn.onclick = showinterPrompt;
        }
        if (!langchainBtn) {
            langchainBtn = document.createElement('button');
            langchainBtn.id = 'langchainBtn';
            langchainBtn.textContent = 'Langchain Prompt';
            langchainBtn.className = 'dev-mode-btn';  // Add class for styling
            xlsxbtn.appendChild(langchainBtn);
            // Add click event to open popup with text file content
            langchainBtn.onclick = showLangPromptPopup;

        }
    } else {
        // Remove buttons if they exist
        if (interpBtn) {
            interpBtn.remove();
        }
        if (langchainBtn) {
            langchainBtn.remove();
        }
    }
}

// Database and Section Dropdown Handling
function connectToDatabase(selectedDatabase) {
    const sectionDropdown = document.getElementById('section-dropdown');
    const connectionStatus = document.getElementById('connection-status');

    // Clear previous options
    sectionDropdown.innerHTML = '<option value="" disabled selected>Select Subject</option>';

    // Update connection status
    connectionStatus.textContent = `Connecting to ${selectedDatabase}...`;
    connectionStatus.style.color = 'orange';

    // Get the appropriate section template based on database selection
    let sectionTemplateId;
    let sections = [];

    if (selectedDatabase === 'GCP') {
        sections = ['Demo']; // Directly specify GCP sections
    } else if (selectedDatabase == 'PostgreSQL-Azure') {
        sections = [
            'Finance', 'Customer Support', 'HR', 'Healthcare',
            'Insurance', 'Inventory', 'Legal', 'Sales'
        ]; // Directly specify PostgreSQL sections
    } else {
        console.error('Unknown database selected:', selectedDatabase);
        return;
    }

    // Add sections to dropdown
    sections.forEach(section => {
        const option = document.createElement('option');
        option.value = section;
        option.textContent = section;
        sectionDropdown.appendChild(option);
    });

    // Enable the dropdown and update status
    sectionDropdown.disabled = false;
    connectionStatus.textContent = `Connected to ${selectedDatabase}`;
    connectionStatus.style.color = 'green';

    // Reset any previous selections
    sectionDropdown.selectedIndex = 0;

    // // Fetch questions for the default section (if needed)
    // if (sections.length > 0) {
    //     fetchQuestions(sections[0]);
    // }
}

// Initialize event listeners for database dropdown
document.addEventListener('DOMContentLoaded', function () {
    // Database dropdown initialization
    const savedDatabase = sessionStorage.getItem('selectedDatabase');
    const savedSection = sessionStorage.getItem('selectedSection');
    if (savedDatabase) {
        document.getElementById('database-dropdown').value = savedDatabase;
        connectToDatabase(savedDatabase);
    }
    
    if (savedSection) {
        document.getElementById('section-dropdown').value = savedSection;
        fetchQuestions(savedSection);
    }

    // Other initializations
    document.getElementById("chat-mic-button").addEventListener("click", toggleRecording);
    document.getElementById("chat_user_query").addEventListener("keyup", function (event) {
        if (event.key === "Enter") sendMessage();
    });

    // Set default tab
    document.getElementsByClassName("tablinks")[0]?.click();
});

// Add this to your DOMContentLoaded event listener
document.getElementById('section-dropdown').addEventListener('change', function() {
    const selectedDatabase = document.getElementById('database-dropdown').value;
    const selectedSection = this.value;
    
    if (selectedDatabase && selectedSection) {
        sessionStorage.setItem('selectedDatabase', selectedDatabase);
        sessionStorage.setItem('selectedSection', selectedSection);
        location.reload();
    }
});

// Your existing sendMessage function with modifications
async function sendMessage() {
    const userQueryInput = document.getElementById("chat_user_query");
    const chatMessages = document.getElementById("chat-messages");
    const typingIndicator = document.getElementById("typing-indicator");
    const queryResultsDiv = document.getElementById('query-results');

    let userMessage = userQueryInput.value.trim();
    if (!userMessage) return;

    // Get selected database and section
    const selectedDatabase = document.getElementById('database-dropdown').value;
    
    // Get current database and section from session storage
    const currentDatabase = sessionStorage.getItem('selectedDatabase');
    const currentSection = sessionStorage.getItem('selectedSection');
    if (selectedDatabase && selectedDatabase !== currentDatabase) {
        // Store the selected database in sessionStorage before reloading
        sessionStorage.setItem('selectedDatabase', selectedDatabase);
        location.reload();
        return;
    }
    const selectedSection = document.getElementById('section-dropdown').value;
    if ((selectedDatabase && selectedDatabase !== currentDatabase) || 
    (selectedSection && selectedSection !== currentSection)) {
    // Store the new selections in sessionStorage
    sessionStorage.setItem('selectedDatabase', selectedDatabase);
    sessionStorage.setItem('selectedSection', selectedSection);
    location.reload();
    return;
}
    // Validate selection
    if (!selectedDatabase || !selectedSection) {
        alert("Please select both a database and a subject area");
        return;
    }

    // Append user message
    chatMessages.innerHTML += `
        <div class="message user-message">
            <div class="message-content">${userMessage}</div>
        </div>
    `;
    userQueryInput.value = "";
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Show typing indicator
    typingIndicator.style.display = "flex";
    queryResultsDiv.style.display = "block";

    try {
        const formData = new FormData();
        formData.append('user_query', userMessage);
        formData.append('section', selectedSection);
        formData.append('database', selectedDatabase);
        console.log(selectedSection) // Add database to form data
        console.log(selectedDatabase)
        const response = await fetch("/submit", { method: "POST", body: formData });

        if (!response.ok) throw new Error("Failed to fetch response");

        const data = await response.json();
        typingIndicator.style.display = "none";

        let botResponse = "";

        if (!data.query) {
            botResponse = data.chat_response || "I couldn't find any insights for this query.";
        } else {
            document.getElementById("sql-query-content").textContent = data.query;
            botResponse = data.chat_response || "Here's what I found:";
        }
        console.log("interprompt: ", data.interprompt)
        document.getElementById("lang-prompt-content").textContent = data.langprompt;
        document.getElementById("interp-prompt-content").textContent = data.interprompt;

        chatMessages.innerHTML += `
            <div class="message ai-message">
                <div class="message-content">
                    LLM Interpretation: ${data.llm_response}<br>
                    Insight: ${botResponse}
                </div>
            </div>
        `;

        chatMessages.scrollTop = chatMessages.scrollHeight;
        if (data.tables) {
            console.log()
            tableName = data.tables[0].table_name;
            loadTableColumns(tableName)
            updatePageContent(data);
        }
    } catch (error) {
        console.error("Error:", error);
        typingIndicator.style.display = "none";
        alert("Error processing request.");
    }
}

// Your existing mic recording function
async function toggleRecording() {
    const micButton = document.getElementById("chat-mic-button");

    if (!isRecording) {
        // Store the original button HTML before changing it
        originalButtonHTML = micButton.innerHTML;

        // Start recording
        micButton.innerHTML = "Recording... (Click to stop)";

        isRecording = true;
        audioChunks = []; // Reset recorded data

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                isRecording = false; // Allow next recording

                if (audioChunks.length === 0) {
                    alert("No audio recorded.");
                    return;
                }

                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const formData = new FormData();
                formData.append("file", audioBlob, "recording.webm");

                try {
                    console.log("Sending audio file to server...");
                    const response = await fetch("/transcribe-audio/", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();
                    console.log("Server Response:", data);

                    if (data.transcription) {
                        document.getElementById("chat_user_query").value = data.transcription;
                    } else {
                        alert("Failed to transcribe audio.");
                    }
                } catch (error) {
                    console.error("Error transcribing audio:", error);
                    alert("An error occurred while transcribing.");
                }

                // Restore the original button HTML (image inside button)
                micButton.innerHTML = originalButtonHTML;
            };

            mediaRecorder.start();
            console.log("Recording started...");
        } catch (error) {
            console.error("Microphone access denied or error:", error);
            alert("Microphone access denied. Please allow microphone permissions.");
            isRecording = false;
        }
    } else {
        // Stop recording
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
            console.log("Recording stopped.");
        }
    }
}


// Initialize mic button listener
document.getElementById("chat-mic-button").addEventListener("click", toggleRecording); async function toggleRecording() {
    const micButton = document.getElementById("chat-mic-button");

    if (!isRecording) {
        // Store the original button HTML before changing it
        originalButtonHTML = micButton.innerHTML;

        // Start recording
        micButton.innerHTML = "Recording... (Click to stop)";

        isRecording = true;
        audioChunks = []; // Reset recorded data

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                isRecording = false; // Allow next recording

                if (audioChunks.length === 0) {
                    alert("No audio recorded.");
                    return;
                }

                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const formData = new FormData();
                formData.append("file", audioBlob, "recording.webm");

                try {
                    console.log("Sending audio file to server...");
                    const response = await fetch("/transcribe-audio/", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();
                    console.log("Server Response:", data);

                    if (data.transcription) {
                        document.getElementById("chat_user_query").value = data.transcription;
                    } else {
                        alert("Failed to transcribe audio.");
                    }
                } catch (error) {
                    console.error("Error transcribing audio:", error);
                    alert("An error occurred while transcribing.");
                }

                // Restore the original button HTML (image inside button)
                micButton.innerHTML = originalButtonHTML;
            };

            mediaRecorder.start();
            console.log("Recording started...");
        } catch (error) {
            console.error("Microphone access denied or error:", error);
            alert("Microphone access denied. Please allow microphone permissions.");
            isRecording = false;
        }
    } else {
        // Stop recording
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
            console.log("Recording stopped.");
        }
    }
}

// Attach event listener to button
document.getElementById("chat-mic-button").addEventListener("click", toggleRecording);
// document.getElementById("section-dropdown")?.addEventListener("change", (event) => {
//     fetchQuestions(event.target.value)
// });

// Event listener for table dropdown change (if it exists)
document.getElementById("table-dropdown")?.addEventListener("change", (event) => {
    document.getElementById("download-button").style.display = event.target.value ? "block" : "none";
});
/**
 *
 */
/**
 * Resets the session state by making a POST request to the backend.
 */
function resetSession() {
    // Show a confirmation dialog first
    const confirmed = confirm("Are you sure you want to reset your session? This will clear all your current data.");
    
    if (!confirmed) return;

    // Show loading state (assuming you have a way to display this)
    showLoadingIndicator("Resetting session...");

    fetch('/reset-session', { method: 'POST' })
        .then(response => {
            if (response.ok) {
                // More friendly success message
                showToastMessage("Session reset successfully! Refreshing your page...", 'success');
                
                // Brief delay before reload to let user see the message
                setTimeout(() => {
                    location.reload();
                }, 1500);
            } else {
                // More detailed error message
                showToastMessage("We couldn't reset your session. Please try again later.", 'error');
            }
        })
        .catch(error => {
            console.error("Error resetting session:", error);
            showToastMessage("A network error occurred. Please check your connection and try again.", 'error');
        })
        .finally(() => {
            hideLoadingIndicator();
        });
}

// Helper functions for UI feedback (you'll need to implement these or use a library)
function showToastMessage(message, type = 'info') {
    // Implement or replace with your preferred notification system
    // Example using a library: Toastify, SweetAlert, etc.
    alert(message); // Fallback - replace with better UI
}

function showLoadingIndicator(message) {
    // Could be a spinner with text, progress bar, etc.
    console.log("Loading: " + message); // Fallback
}

function hideLoadingIndicator() {
    // Hide whatever loading indicator you showed
    console.log("Loading complete"); // Fallback
}

async function fetchQuestions(selectedSection) {
    const questionDropdown = document.getElementById("faq-questions"); // Get datalist
    questionDropdown.innerHTML = ''; // Clear previous options

    if (selectedSection) {
        try {
            const response = await fetch(`/get_questions?subject=${selectedSection}`);
            const data = await response.json();

            if (data.questions && data.questions.length > 0) {
                data.questions.forEach(question => {
                    const option = document.createElement("option");
                    option.value = question; // Set the value directly
                    questionDropdown.appendChild(option);
                });
            } else {
                console.warn(`No questions found for section: ${selectedSection}`);
            }
        } catch (error) {
            console.error("Error fetching questions:", error);
        }
    }
}
async function submitFeedback(tableName, feedbackType) {
    const userQueryInput = document.getElementById("chat_user_query");
    const userQuery = userQueryInput.value;
    const sqlQueryDisplay = document.getElementById("sql_query_display");
    const sqlQuery = sqlQueryDisplay.textContent.replace('SQL Query:', '').trim();

    try {
        const response = await fetch("/submit_feedback", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                table_name: tableName,
                feedback_type: feedbackType,
                user_query: userQuery,
                sql_query: sqlQuery
            }),
        });

        const data = await response.json();
        const feedbackMessageElement = document.getElementById(`${tableName}_feedback_message`);
        feedbackMessageElement.textContent = data.message; // Display server's message
        feedbackMessageElement.style.color = (data.success) ? 'green' : 'red'; // Color code the message
    } catch (error) {
        console.error("Error submitting feedback:", error);
        const feedbackMessageElement = document.getElementById(`${tableName}_feedback_message`);
        feedbackMessageElement.textContent = "Failed to submit feedback.";
        feedbackMessageElement.style.color = 'red';
    }
}
// // Ensure that the function is called when the section is selected
// document.getElementById("section-dropdown")?.addEventListener("change", (event) => {
//     fetchQuestions(event.target.value);
// });

/**
 *
 */
function clearQuery() {
    const userQueryInput = document.getElementById("chat_user_query"); // changed id
    userQueryInput.value = ""

}
/**
 *
 */
function chooseExampleQuestion() {
    const questionDropdown = document.getElementById("questions-dropdown");
    const selectedQuestion = questionDropdown.options[questionDropdown.selectedIndex].text;
    if (!selectedQuestion || selectedQuestion === "Select a Question") {
        alert("Please select a question.");
        return;
    }
    const userQueryInput = document.getElementById("chat_user_query"); // changed id
    userQueryInput.value = selectedQuestion;
}
/**
 *
 */
function updatePageContent(data) {
    const userQueryDisplay = document.getElementById("user_query_display");
    const sqlQueryContent = document.getElementById("sql-query-content"); // Get the modal content
    const tablesContainer = document.getElementById("tables_container");
    const xlsxbtn = document.getElementById("xlsx-btn"); // Excel button container
    const emailbtn = document.getElementById("email-btn"); // Excel button container
    const faqbtn = document.getElementById("add-to-faqs-btn");
    // Update user query text
    userQueryDisplay.querySelector('span').textContent = data.user_query || "";

    // Clear and update tables container
    tablesContainer.innerHTML = "";
    xlsxbtn.innerHTML = ""; // Clear Excel button container before adding new buttons
    if (data.tables && data.tables.length > 0) {
        data.tables.forEach((table) => {
            const tableWrapper = document.createElement("div");

            tableWrapper.innerHTML = `
                <div id="${table.table_name}_table">${table.table_html}</div>
                <div id="${table.table_name}_pagination"></div>
                <div id="${table.table_name}_error"></div>
                <div class="feedback-section">
                    <button class="like-button" data-table="${table.table_name}" onclick="submitFeedback('${table.table_name}', 'like')">Like</button>
                    <button class="dislike-button" data-table="${table.table_name}" onclick="submitFeedback('${table.table_name}', 'dislike')">Dislike</button>
                    <span id="${table.table_name}_feedback_message"></span>
                </div>
            `;

            tablesContainer.appendChild(tableWrapper);

            // Create "Download Excel" button with spacing
            const downloadButton = document.createElement("button");
            downloadButton.id = `download-button-${table.table_name}`;
            downloadButton.className = "download-btn";
            downloadButton.innerHTML = `<img src="static/excel.png" alt="xlsx" class="excel-icon"> Download Excel`;
            downloadButton.onclick = () => downloadSpecificTable(table.table_name);

            xlsxbtn.appendChild(downloadButton);
            // Add pagination
            updatePaginationLinks(
                table.table_name,
                table.pagination.current_page,
                table.pagination.total_pages,
                table.pagination.records_per_page
            );

        });
    } else {
        tablesContainer.innerHTML = "<p>No tables to display.</p>";
    }
   // Add copy button in top-right of popup
   const copyButton = document.createElement('button');
   copyButton.innerHTML = '<i class="fas fa-copy"></i>';
   copyButton.className = 'copy-btn-popup';
   copyButton.addEventListener('click', () => {
       const sqlQueryText = document.getElementById("sql-query-content").textContent;
       navigator.clipboard.writeText(sqlQueryText)
           .then(() => {
               alert('SQL query copied to clipboard!');
           })
           .catch(err => {
               console.error('Failed to copy: ', err);
               alert('Failed to copy SQL query to clipboard.');
           });
   });

   // Ensure this is inside the modal
   sqlQueryContent.parentNode.appendChild(copyButton);

    // Add the "View SQL Query" button BELOW the Download Excel button
    if (data.query) {
        sqlQueryContent.textContent = data.query;

        // Create "View SQL Query" button dynamically
        const viewQueryBtn = document.createElement("button");
        viewQueryBtn.textContent = "SQL Query";
        viewQueryBtn.id = "view-sql-query-btn";
        viewQueryBtn.onclick = showSQLQueryPopup;
        viewQueryBtn.style.display = "block"; // Ensure button appears in a new line
        const faqBtn = document.createElement("button");
        faqBtn.textContent = "Add to FAQs";
        faqBtn.id = "add-to-faqs-btn";
        faqBtn.onclick = addToFAQs;
        faqBtn.style.display = "block"; // Ensure button appears in a new line
        const emailbtn = document.createElement("button");
        emailbtn.id = "send-email-btn";

        emailbtn.textContent = "Send Email";

        emailbtn.style.display = "block";
        xlsxbtn.appendChild(viewQueryBtn); // Append below the Excel download button
        xlsxbtn.appendChild(faqBtn);
        xlsxbtn.appendChild(emailbtn) // Append below the Excel download button
    } else {
        sqlQueryContent.textContent = "No SQL query available.";
    }
}
/**
 *
 */
function addToFAQs() {
    let userQuery = document.querySelector("#user_query_display span").innerText;

    if (!userQuery.trim()) {
        document.getElementById("faq-message").innerText = "Query cannot be empty!";
        return;
    }

    fetch('/add_to_faqs', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: userQuery })
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById("faq-message").innerText = data.message;
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById("faq-message").innerText = "Failed to add query to FAQs!";
        });
}
/**
 * @param {any} tableName
 */
function downloadSpecificTable(tableName) {
    // Corrected: Using template literals to construct the URL
    const downloadUrl = `/download-table?table_name=${encodeURIComponent(tableName)}`;
    window.location.href = downloadUrl;
}
/**
 *
 */
function updatePaginationLinks(tableName, currentPage, totalPages, recordsPerPage) {
    const paginationDiv = document.getElementById(`${tableName}_pagination`);
    if (!paginationDiv) return;

    paginationDiv.innerHTML = "";
    const paginationList = document.createElement("ul");
    paginationList.className = "pagination";

    // Calculate start and end pages to display
    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, startPage + 4);

    // Ensure at most 5 page numbers are shown
    if (endPage - startPage < 4) {
        startPage = Math.max(1, endPage - 4);
    }

    // Previous Button
    const prevLi = document.createElement("li");
    prevLi.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
    prevLi.innerHTML = `<a href="javascript:void(0);" onclick="${currentPage > 1 ? `changePage('${tableName}', ${currentPage - 1}, ${recordsPerPage})` : 'return false;'}" class="page-link">« Prev</a>`;
    paginationList.appendChild(prevLi);

    // Show "1 ..." if the startPage is greater than 1
    if (startPage > 1) {
        const firstPageLi = document.createElement("li");
        firstPageLi.className = "page-item";
        firstPageLi.innerHTML = `<a href="javascript:void(0);" onclick="changePage('${tableName}', 1, ${recordsPerPage})" class="page-link">1</a>`;
        paginationList.appendChild(firstPageLi);

        if (startPage > 2) {
            const dotsLi = document.createElement("li");
            dotsLi.className = "page-item disabled";
            dotsLi.innerHTML = `<span class="page-link">...</span>`;
            paginationList.appendChild(dotsLi);
        }
    }

    // Page Numbers
    for (let page = startPage; page <= endPage; page++) {
        const pageLi = document.createElement("li");
        pageLi.className = `page-item ${page === currentPage ? 'active' : ''}`;
        pageLi.innerHTML = `<a href="javascript:void(0);" onclick="changePage('${tableName}', ${page}, ${recordsPerPage})" class="page-link">${page}</a>`;
        paginationList.appendChild(pageLi);
    }

    // Show "... totalPages" if endPage is less than totalPages
    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            const dotsLi = document.createElement("li");
            dotsLi.className = "page-item disabled";
            dotsLi.innerHTML = `<span class="page-link">...</span>`;
            paginationList.appendChild(dotsLi);
        }
        const lastPageLi = document.createElement("li");
        lastPageLi.className = "page-item";
        lastPageLi.innerHTML = `<a href="javascript:void(0);" onclick="changePage('${tableName}', ${totalPages}, ${recordsPerPage})" class="page-link">${totalPages}</a>`;
        paginationList.appendChild(lastPageLi);
    }

    // Next Button
    const nextLi = document.createElement("li");
    nextLi.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
    nextLi.innerHTML = `<a href="javascript:void(0);" onclick="${currentPage < totalPages ? `changePage('${tableName}', ${currentPage + 1}, ${recordsPerPage})` : 'return false;'}" class="page-link">Next »</a>`;
    paginationList.appendChild(nextLi);

    paginationDiv.appendChild(paginationList);
}
// Function to show SQL query in popup
function showSQLQueryPopup() {
    const sqlQueryText = document.getElementById("sql-query-content").textContent;

    if (!sqlQueryText.trim()) {
        alert("No SQL query available.");
        return;
    }

    document.getElementById("sql-query-content").textContent = sqlQueryText;
    document.getElementById("sql-query-popup").style.display = "flex";
    Prism.highlightAll(); // Apply syntax highlighting
}

// Function to close the popup
function closeSQLQueryPopup() {
    document.getElementById("sql-query-popup").style.display = "none";
}
function showLangPromptPopup() {
    document.getElementById("lang-prompt-popup").style.display = "flex";
}


// Function to close the popup
function closepromptPopup() {
    document.getElementById("lang-prompt-popup").style.display = "none";
}
function showinterPrompt() {
    document.getElementById("interp-prompt-popup").style.display = "flex";
}

function closeinterpromptPopup() {
    document.getElementById("interp-prompt-popup").style.display = "none";
}