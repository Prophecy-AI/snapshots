
using DataFrames
using MachineLearning
using JSON

function read_data(file_name)
    f = open(file_name)
    json = JSON.parse(readall(f))
    close(f)

    colnames = keys(json[1])
    columns  = Any[[json[i][name] for i=1:length(json)] for name=colnames]
    DataFrame(columns, Symbol[name for name=colnames])
end

train = read_data("../input/train.json")
test  = read_data("../input/test.json")

println(@sprintf("There are %d rows in the training set", nrow(train)))
println(@sprintf("There are %d rows in the test set", nrow(test)))

feature_names = Symbol["requester_account_age_in_days_at_request",
                       "requester_days_since_first_post_on_raop_at_request",
                       "requester_number_of_comments_at_request",
                       "requester_number_of_comments_in_raop_at_request",
                       "requester_number_of_posts_at_request",
                       "requester_number_of_posts_on_raop_at_request",
                       "requester_number_of_subreddits_at_request",
                       "requester_upvotes_minus_downvotes_at_request",
                       "requester_upvotes_plus_downvotes_at_request",
                       "unix_timestamp_of_request_utc"]

for feature = feature_names
    train[feature] = float64(train[feature])
    test[feature]  = float64(test[feature])
end

columns_to_keep = cat(1, feature_names, [:requester_received_pizza])

rf = fit(train[columns_to_keep], :requester_received_pizza, classification_forest_options(num_trees=200, display=true))
println("")
println(rf)
println("")
predictions = predict_probs(rf, test)[:,2]
submission = DataFrame(request_id=test[:request_id], requester_received_pizza=predictions)
writetable("simple_julia_benchmark.csv", submission)
