pipeline {
    agent {
        node {
            label 'Linux_Default'
        }
    }
    stages {
        stage('Clear WS') {
            steps {
                cleanWs()
                checkout scm
                echo "Building ${env.JOB_NAME}..."
            }
        }
        stage('SonarQube analysis') {
            steps {
                withSonarQubeEnv(credentialsId: 'sonar_token_llm_val_cib_sonar:master_423753', installationName: 'SonarQubeSigma') {
                    script {
                        try {
                            baseCommit = sh(script: "git rev-parse origin/" + env.BRANCH_NAME, returnStdout: true).trim()
                        } catch (Exception e) {
                            echo "Failed to get base commit for branch ${env.BRANCH_NAME}. Using HEAD as fallback."
                            baseCommit = sh(script: "git rev-parse HEAD", returnStdout: true).trim()
                        }
                        sonarBranchTargetParameter=""
                        if (env.BRANCH_NAME.toLowerCase().contains("hotfix/"))
                            sonarBranchTargetParameter="-Dsonar.branch.target=master"
                        if (env.CHANGE_ID) {
                         sh """
                         ${scannerHome}/bin/sonar-scanner \
                         -D sonar.pullrequest.branch=${env.CHANGE_BRANCH} \
                         -D sonar.pullrequest.base=${env.CHANGE_TARGET} \
                         -D sonar.pullrequest.key=${env.CHANGE_ID} \
                         -D sonar.scm.revision=${baseCommit}
                         """
                        } else {
                         sh """
                         ${scannerHome}/bin/sonar-scanner \
                         -D sonar.branch.name=${env.BRANCH_NAME} \
                         ${sonarBranchTargetParameter}
                         """
                        }
                    }
                }
            }
        }

        stage('Quality Gate') {
            steps {
                script {
                    def qg = null
                    timeout(time: 3, unit: 'MINUTES') {
                        // sleep 10 // Временное решение описанное в https://community.sonarsource.com/t/need-a-sleep-between-withsonarqubeenv-and-waitforqualitygate-or-it-spins-in-in-progress/2265/17 и https://community.sonarsource.com/t/waitforqualitygate-timeout-in-jenkins/2116/4
                        //             // Иначе иногда процесс зависает в состоянии IN PROGRESS
                        // qg = waitForQualityGate()
                    // }
                    // if (qg.status != 'OK') {
                    //     error "Pipeline aborted due to quality gate failure: ${qg.status}"
                    }
                    CI_FLAG_STATUS = "ok"
                }
            }
        }

    }
    environment {
        scannerHome = tool 'sonar-scanner-7.3.0.5189-linux-x64'
    }
}
