const loadingDiv = document.getElementById('loading');
let tableName;
let isRecording = false;
let mediaRecorder;
let audioChunks = [];
let originalButtonHTML = ""; // Store the original button HTML

// Database and Section Dropdown Handling
function connectToDatabase(selectedDatabase) {
    const sectionDropdown = document.getElementById('section-dropdown');
    const connectionStatus = document.getElementById('connection-status');
    
    // Clear previous options
    sectionDropdown.innerHTML = '<option value="" disabled selected>Select Subject</option>';
    sectionDropdown.disabled = false;
    
    // Update connection status
    connectionStatus.textContent = `Connected to ${selectedDatabase}`;
    connectionStatus.style.color = 'green';
    
    // Get the appropriate sections based on selected database
    let sections = [];
    if (selectedDatabase === 'GCP') {
        sections = Array.from(document.getElementById('gcp-sections').querySelectorAll('option'));
    } else if (selectedDatabase === 'PostgreSQL-Azure') {
        sections = Array.from(document.getElementById('postgresql-sections').querySelectorAll('option'));
    }
    
    // Populate the section dropdown
    sections.forEach(option => {
        sectionDropdown.appendChild(option.cloneNode(true));
    });
    
    // Store the selected database in session
    sessionStorage.setItem('selectedDatabase', selectedDatabase);
    
    // Reset any previous selections
    sectionDropdown.selectedIndex = 0;
    
    // Fetch questions for the first section (if needed)
    if (sections.length > 0) {
        fetchQuestions(sections[0].value);
    }
}

async function loadTableColumns(table_name) {
    console.log("Loading columns for table:", table_name);
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
    if (event.key === "Enter") {
        event.preventDefault();
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
            chartContainer.innerHTML = "";
            const chartDiv = document.createElement("div");
            chartContainer.appendChild(chartDiv);

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

async function sendMessage() {
    const userQueryInput = document.getElementById("chat_user_query");
    const chatMessages = document.getElementById("chat-messages");
    const typingIndicator = document.getElementById("typing-indicator");
    const queryResultsDiv = document.getElementById('query-results');

    let userMessage = userQueryInput.value.trim();
    if (!userMessage) return;

    // Get selected database and section
    const selectedDatabase = document.getElementById('database-dropdown').value;
    const selectedSection = document.getElementById('section-dropdown').value;

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

        chatMessages.innerHTML += `
            <div class="message ai-message">
                <div class="message-content">
                    ${botResponse}
                </div>
            </div>
        `;

        chatMessages.scrollTop = chatMessages.scrollHeight;
        if (data.tables) {
            tableName = data.tables[0].table_name;
            loadTableColumns(tableName);
            updatePageContent(data);
        }
    } catch (error) {
        console.error("Error:", error);
        typingIndicator.style.display = "none";
        alert("Error processing request.");
    }
}

async function toggleRecording() {
    const micButton = document.getElementById("chat-mic-button");

    if (!isRecording) {
        originalButtonHTML = micButton.innerHTML;
        micButton.innerHTML = "Recording... (Click to stop)";
        isRecording = true;
        audioChunks = [];

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                isRecording = false;

                if (audioChunks.length === 0) {
                    alert("No audio recorded.");
                    return;
                }

                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const formData = new FormData();
                formData.append("file", audioBlob, "recording.webm");

                try {
                    const response = await fetch("/transcribe-audio/", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();
                    if (data.transcription) {
                        document.getElementById("chat_user_query").value = data.transcription;
                    } else {
                        alert("Failed to transcribe audio.");
                    }
                } catch (error) {
                    console.error("Error transcribing audio:", error);
                    alert("An error occurred while transcribing.");
                }

                micButton.innerHTML = originalButtonHTML;
            };

            mediaRecorder.start();
        } catch (error) {
            console.error("Microphone access denied or error:", error);
            alert("Microphone access denied. Please allow microphone permissions.");
            isRecording = false;
        }
    } else {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
        }
    }
}

function resetSession() {
    fetch('/reset-session', { method: 'POST' })
        .then(response => {
            if (response.ok) {
                alert("Session reset successfully!");
                location.reload();
            } else {
                alert("Failed to reset session.");
            }
        })
        .catch(error => console.error("Error resetting session:", error));
}

async function fetchQuestions(selectedSection) {
    const questionDropdown = document.getElementById("faq-questions");
    questionDropdown.innerHTML = '';

    if (selectedSection) {
        try {
            const response = await fetch(`/get_questions?subject=${selectedSection}`);
            const data = await response.json();

            if (data.questions && data.questions.length > 0) {
                data.questions.forEach(question => {
                    const option = document.createElement("option");
                    option.value = question;
                    questionDropdown.appendChild(option);
                });
            }
        } catch (error) {
            console.error("Error fetching questions:", error);
        }
    }
}

function clearQuery() {
    document.getElementById("chat_user_query").value = "";
}

function chooseExampleQuestion() {
    const questionDropdown = document.getElementById("questions-dropdown");
    const selectedQuestion = questionDropdown.options[questionDropdown.selectedIndex].text;
    if (!selectedQuestion || selectedQuestion === "Select a Question") {
        alert("Please select a question.");
        return;
    }
    document.getElementById("chat_user_query").value = selectedQuestion;
}

function updatePageContent(data) {
    const userQueryDisplay = document.getElementById("user_query_display");
    const sqlQueryContent = document.getElementById("sql-query-content");
    const tablesContainer = document.getElementById("tables_container");
    const xlsxbtn = document.getElementById("xlsx-btn");

    userQueryDisplay.querySelector('span').textContent = data.user_query || "";
    tablesContainer.innerHTML = "";
    xlsxbtn.innerHTML = "";

    if (data.tables && data.tables.length > 0) {
        data.tables.forEach((table) => {
            const tableWrapper = document.createElement("div");
            tableWrapper.innerHTML = `
                <div id="${table.table_name}_table">${table.table_html}</div>
                <div id="${table.table_name}_pagination"></div>
                <div id="${table.table_name}_error"></div>
            `;
            tablesContainer.appendChild(tableWrapper);

            const downloadButton = document.createElement("button");
            downloadButton.id = `download-button-${table.table_name}`;
            downloadButton.className = "download-btn";
            downloadButton.innerHTML = `<img src="static/excel.png" alt="xlsx" class="excel-icon"> Download Excel`;
            downloadButton.onclick = () => downloadSpecificTable(table.table_name);
            xlsxbtn.appendChild(downloadButton);

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

    if (data.query) {
        sqlQueryContent.textContent = data.query;

        const viewQueryBtn = document.createElement("button");
        viewQueryBtn.textContent = "SQL Query";
        viewQueryBtn.id = "view-sql-query-btn";
        viewQueryBtn.onclick = showSQLQueryPopup;
        viewQueryBtn.style.display = "block";
        
        const faqBtn = document.createElement("button");
        faqBtn.textContent = "Add to FAQs";
        faqBtn.id = "add-to-faqs-btn";
        faqBtn.onclick = addToFAQs;
        faqBtn.style.display = "block";

        xlsxbtn.appendChild(viewQueryBtn);
        xlsxbtn.appendChild(faqBtn);
    } else {
        sqlQueryContent.textContent = "No SQL query available.";
    }
}

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

function downloadSpecificTable(tableName) {
    const downloadUrl = `/download-table?table_name=${encodeURIComponent(tableName)}`;
    window.location.href = downloadUrl;
}

function updatePaginationLinks(tableName, currentPage, totalPages, recordsPerPage) {
    const paginationDiv = document.getElementById(`${tableName}_pagination`);
    if (!paginationDiv) return;

    paginationDiv.innerHTML = "";
    const paginationList = document.createElement("ul");
    paginationList.className = "pagination";

    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, startPage + 4);

    if (endPage - startPage < 4) {
        startPage = Math.max(1, endPage - 4);
    }

    // Previous Button
    const prevLi = document.createElement("li");
    prevLi.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
    prevLi.innerHTML = `<a href="javascript:void(0);" onclick="changePage('${tableName}', ${currentPage - 1}, ${recordsPerPage})" class="page-link">« Prev</a>`;
    paginationList.appendChild(prevLi);

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

    for (let page = startPage; page <= endPage; page++) {
        const pageLi = document.createElement("li");
        pageLi.className = `page-item ${page === currentPage ? 'active' : ''}`;
        pageLi.innerHTML = `<a href="javascript:void(0);" onclick="changePage('${tableName}', ${page}, ${recordsPerPage})" class="page-link">${page}</a>`;
        paginationList.appendChild(pageLi);
    }

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

    const nextLi = document.createElement("li");
    nextLi.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
    nextLi.innerHTML = `<a href="javascript:void(0);" onclick="changePage('${tableName}', ${currentPage + 1}, ${recordsPerPage})" class="page-link">Next »</a>`;
    paginationList.appendChild(nextLi);

    paginationDiv.appendChild(paginationList);
}

function showSQLQueryPopup() {
    const sqlQueryText = document.getElementById("sql-query-content").textContent;

    if (!sqlQueryText.trim()) {
        alert("No SQL query available.");
        return;
    }

    document.getElementById("sql-query-content").textContent = sqlQueryText;
    document.getElementById("sql-query-popup").style.display = "flex";
    Prism.highlightAll();
}

function closeSQLQueryPopup() {
    document.getElementById("sql-query-popup").style.display = "none";
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Check for previously selected database
    const savedDb = sessionStorage.getItem('selectedDatabase');
    if (savedDb) {
        const dbDropdown = document.getElementById('database-dropdown');
        for (let i = 0; i < dbDropdown.options.length; i++) {
            if (dbDropdown.options[i].value === savedDb) {
                dbDropdown.selectedIndex = i;
                connectToDatabase(savedDb);
                break;
            }
        }
    }

    // Set initial connection status
    document.getElementById('connection-status').textContent = "Select a database";

    // Initialize event listeners
    document.getElementById("chat-mic-button").addEventListener("click", toggleRecording);
    document.getElementById("chat_user_query").addEventListener("keyup", function(event) {
        if (event.key === "Enter") sendMessage();
    });
    
    document.getElementById('database-dropdown').addEventListener('change', function() {
        connectToDatabase(this.value);
    });
    
    document.getElementById('section-dropdown').addEventListener('change', function() {
        if (this.value) {
            fetchQuestions(this.value);
        }
    });

    // Set default tab
    document.getElementsByClassName("tablinks")[0]?.click();
});