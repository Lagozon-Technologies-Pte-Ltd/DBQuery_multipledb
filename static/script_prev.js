
const loadingDiv = document.getElementById('loading');
let tableName;
let isRecording = false;
let mediaRecorder;
let audioChunks = [];
let originalButtonHTML = ""; // Store the original button HTML
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


async function sendMessage() {
    const userQueryInput = document.getElementById("chat_user_query");
    const chatMessages = document.getElementById("chat-messages");
    const typingIndicator = document.getElementById("typing-indicator");
    const queryResultsDiv = document.getElementById('query-results');

    let userMessage = userQueryInput.value.trim();
    if (!userMessage) return;

    // Append user message
    chatMessages.innerHTML += `
        <div class="message user-message">
            <div class="message-content">${userMessage}</div>
        </div>
    `;
    userQueryInput.value = ""; // Clear input field
    chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to bottom

    // Show typing indicator
    typingIndicator.style.display = "flex";
    queryResultsDiv.style.display = "block";

    try {
        const formData = new FormData();
        formData.append('user_query', userMessage);
        formData.append('section', document.getElementById('section-dropdown').value);

        const response = await fetch("/submit", { method: "POST", body: formData });

        if (!response.ok) throw new Error("Failed to fetch response");

        const data = await response.json();
        typingIndicator.style.display = "none";

        let botResponse = "";

        // **If it's an insight message (NO SQL Query), handle it separately**
        if (!data.query) {
            botResponse = data.chat_response || "I couldn't find any insights for this query.";
        } else {
            // **If it's an SQL Query, handle separately**
            document.getElementById("sql-query-content").textContent = data.query;
            botResponse = data.chat_response || "Here's what I found:";
        }

        chatMessages.innerHTML += `
            <div class="message ai-message">
                <div class="message-content">
                    ${botResponse}
                </div>
            </div>
        `;

        // Scroll chat to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        if (data.tables) {
            tableName = data.tables[0].table_name;
            loadTableColumns(tableName)
            updatePageContent(data);


        }
        // Update table, visualization, and query details if applicable
    } catch (error) {
        console.error("Error:", error);
        typingIndicator.style.display = "none";
        alert("Error processing request.");
    }
}
document.getElementById("chat-mic-button").addEventListener("click", toggleRecording);
// Handle Mic Recording (Modify toggleRecording function)
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
    fetch('/reset-session', { method: 'POST' })
        .then(response => {
            if (response.ok) {
                alert("Session reset successfully!");
                // Optionally, reload the page to ensure all UI elements are reset
                location.reload();
            } else {
                alert("Failed to reset session.");
            }
        })
        .catch(error => console.error("Error resetting session:", error));
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
    const faqbtn =  document.getElementById("add-to-faqs-btn");
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

        xlsxbtn.appendChild(viewQueryBtn); // Append below the Excel download button
        xlsxbtn.appendChild(faqBtn); // Append below the Excel download button
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
    prevLi.innerHTML = `<a href="javascript:void(0);" onclick="changePage('${tableName}', ${currentPage - 1}, ${recordsPerPage})" class="page-link">« Prev</a>`;
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
    nextLi.innerHTML = `<a href="javascript:void(0);" onclick="changePage('${tableName}', ${currentPage + 1}, ${recordsPerPage})" class="page-link">Next »</a>`;
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
