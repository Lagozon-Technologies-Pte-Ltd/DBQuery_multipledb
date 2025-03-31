document.addEventListener("DOMContentLoaded", function () {
    const userSection = document.getElementById("user-section");
    const adminSection = document.getElementById("admin-section");
    const queryResults = document.getElementById("query-results");



});
let tableName;
let isRecording = false;
let mediaRecorder;
let audioChunks = [];
let originalButtonHTML = ""; // Store the original button HTML

async function toggleRecording() {
    const micButton = document.getElementById("mic-button");

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
                        document.getElementById("user_query").value = data.transcription;
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
document.getElementById("mic-button").addEventListener("click", toggleRecording);



document.getElementById("table-dropdown")?.addEventListener("change", (event) => {
    document.getElementById("download-button").style.display = event.target.value ? "block" : "none";
});


async function fetchTables(selectedSection) {
    
    if (selectedSection) {
        try {
            // Corrected: Using template literals to construct the URL
            const response = await fetch(`/get-tables?selected_section=${selectedSection}`);
            const data = await response.json();
            if (data.tables && data.tables.length > 0) {
                data.tables.forEach(table => {
                    table = data.tables[0]
                    return table
                  
                });
            } else {
                alert("No tables found for the selected section.");
            }
        } catch (error) {
            console.error("Error fetching tables:", error);
            alert("An error occurred while fetching tables.");
        }
    }
}


async function fetchQuestions(selectedSection) {
    const questionDropdown = document.getElementById("questions-dropdown");
    questionDropdown.innerHTML = '<option value="">Select a Question</option>';
    if (selectedSection) {
        try {
            // Corrected: Using template literals to construct the URL
            const response = await fetch(`/get_questions?subject=${selectedSection}`);
            const data = await response.json();
            if (data.questions && data.questions.length > 0) {
                data.questions.forEach(question => {
                    const option = document.createElement("option");
                    option.value = question;
                    option.textContent = question;
                    questionDropdown.appendChild(option);
                });
            } else {
                alert(`No questions found for the selected ${selectedSection} section.`);
            }
        } catch (error) {
            console.error("Error fetching questions:", error);
            alert("An error occurred while fetching questions.");
        }
    }
}


function clearQuery(){
    const userQueryInput = document.getElementById("user_query");
    userQueryInput.value = ""

}

function chooseExampleQuestion() {
    const questionDropdown = document.getElementById("questions-dropdown");
    const selectedQuestion = questionDropdown.options[questionDropdown.selectedIndex].text;
    if (!selectedQuestion || selectedQuestion === "Select a Question") {
        alert("Please select a question.");
        return;
    }
    const userQueryInput = document.getElementById("user_query");
    userQueryInput.value = selectedQuestion;
}

function handleSubmit(event) {
    event.preventDefault(); // Prevent default form submission
    document.getElementById("loading").style.display = "block";

    const userQueryInput = document.getElementById("user_query");
    const promptInput = document.getElementById("prompt");

    // Ensure the prompt is set to the manually entered query if available
    if (userQueryInput.value.trim() !== "") {
        promptInput.value = userQueryInput.value;
    }
    fetch("/submit", {
        method: "POST",
        body: new FormData(event.target),
    })
        .then((response) => response.json())
        .then((data) => {
            document.getElementById("loading").style.display = "none";
            updatePageContent(data);

            // Show the query results
            const queryResults = document.getElementById("query-results");
            queryResults.style.display = "block";

            // Show the visualization section only if there are tables
            const visualizationSection = document.getElementById("chart-visualization");
            if (data.tables && data.tables.length > 0) {
                tableName = data.tables[0].table_name; 
                console.log(tableName)
                visualizationSection.style.display = "block";
                loadTableColumns(tableName);

            } else {
                visualizationSection.style.display = "none";
            }
        })
        .catch((error) => {
            document.getElementById("loading").style.display = "none";
            alert("Error processing request.");
        });
}


function updatePageContent(data) {
    const userQuery = document.querySelector("#user_query_display");
    const sqlQuery = document.querySelector("#sql_query_display");
    const tablesContainer = document.querySelector("#tables_container");
    const xlsxbtn = document.querySelector("#xlsx-btn");

    // Corrected: Using template literals for textContent
    if (data.user_query) userQuery.textContent = `Query Asked: ${data.user_query}`;
    if (data.query) sqlQuery.innerHTML= `SQL Query: <br><br>${data.query}`;

    tablesContainer.innerHTML = ""; // Clear previous tables
    xlsxbtn.innerHTML = "";
    if (data.tables) {
        data.tables.forEach((table) => {
            // Corrected: Using template literals for HTML
            const tableHtml = `
        <div id="${table.table_name}_table">${table.table_html}</div>
        <div id="${table.table_name}_pagination"></div>
        
        <div id="${table.table_name}_error"></div>
                `;
            const xlsxcontent = `<button id="download-button-${table.table_name}" class="download-btn" 
                onclick="downloadSpecificTable('${table.table_name}')"><img src="static/excel.png" alt="xlsx" class="excel-icon">
                Download Table as Excel</button >`;
            tablesContainer.insertAdjacentHTML("beforeend", tableHtml);
            xlsxbtn.insertAdjacentHTML("beforeend",xlsxcontent)
            // Add pagination
            updatePaginationLinks(
                table.table_name,
                table.pagination.current_page,
                table.pagination.total_pages,
                table.pagination.records_per_page
            );
        });
    }
}

// Function for downloading a specific table
function downloadSpecificTable(tableName) {
    // Corrected: Using template literals to construct the URL
    const downloadUrl = `/download-table?table_name=${encodeURIComponent(tableName)}`;
    window.location.href = downloadUrl;
}

function updatePaginationLinks(tableName, currentPage, totalPages, recordsPerPage) {
    const paginationDiv = document.getElementById(`${tableName}_pagination`);
    if (!paginationDiv) return;

    paginationDiv.innerHTML = "";

    const paginationList = document.createElement("ul");
    paginationList.className = "pagination";

    const prevLi = document.createElement("li");
    prevLi.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
    prevLi.innerHTML = `<a href="javascript:void(0);" onclick="changePage('${tableName}', ${currentPage - 1}, ${recordsPerPage})" class="page-link">« Prev</a>`;
    paginationList.appendChild(prevLi);

    for (let page = 1; page <= totalPages; page++) {
        const pageLi = document.createElement("li");
        pageLi.className = `page-item ${page === currentPage ? 'active' : ''}`;
        pageLi.innerHTML = `<a href="javascript:void(0);" onclick="changePage('${tableName}', ${page}, ${recordsPerPage})" class="page-link">${page}</a>`;
        paginationList.appendChild(pageLi);
    }

    const nextLi = document.createElement("li");
    nextLi.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
    nextLi.innerHTML = `<a href="javascript:void(0);" onclick="changePage('${tableName}', ${currentPage + 1}, ${recordsPerPage})" class="page-link">Next »</a>`;
    paginationList.appendChild(nextLi);

    paginationDiv.appendChild(paginationList);
}
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
async function loadTableColumns(table_name) {
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
